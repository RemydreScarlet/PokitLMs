#include "pokitlms/storage/model_file.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pokitlms::storage {

class ModelFile::Impl {
public:
    explicit Impl(std::filesystem::path source) : path(std::move(source)) {
#if defined(_WIN32)
        stream.open(path, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open model file: " + path.string());
        stream.seekg(0, std::ios::end);
        const auto end = stream.tellg();
        if (end < 0) throw std::runtime_error("cannot determine model file size: " + path.string());
        file_size = static_cast<std::uint64_t>(end);
#else
        int flags = O_RDONLY;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        fd = ::open(path.c_str(), flags);
        if (fd < 0) throw std::runtime_error("cannot open model file: " + path.string());
        struct stat info {};
        if (::fstat(fd, &info) != 0 || info.st_size < 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("cannot determine model file size: " + path.string());
        }
        file_size = static_cast<std::uint64_t>(info.st_size);
#endif
    }

    ~Impl() {
#if !defined(_WIN32)
        if (fd >= 0) ::close(fd);
#endif
    }

    void read_into(std::uint64_t offset, std::span<std::byte> destination,
                   bool discard_cache = false) const {
        if (offset > file_size || destination.size() > file_size - offset) {
            throw std::out_of_range("model file read exceeds file bounds");
        }
        if (destination.empty()) return;
#if defined(_WIN32)
        if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::out_of_range("model file read exceeds stream I/O limits");
        }
        std::lock_guard lock(mutex);
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!stream || !stream.read(reinterpret_cast<char*>(destination.data()),
                                    static_cast<std::streamsize>(destination.size()))) {
            throw std::runtime_error("failed to read model file: " + path.string());
        }
#else
        if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::out_of_range("model file offset exceeds platform I/O limits");
        }
        std::size_t completed = 0;
        constexpr auto max_read = static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
        while (completed < destination.size()) {
            const auto request = std::min(destination.size() - completed, max_read);
            const auto count = ::pread(fd, destination.data() + completed, request,
                static_cast<off_t>(offset + completed));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("failed to read model file: " + path.string());
            completed += static_cast<std::size_t>(count);
        }
        if (discard_cache) {
            const long page_size = ::sysconf(_SC_PAGESIZE);
            if (page_size > 0) {
                const auto page = static_cast<std::uint64_t>(page_size);
                const auto begin = offset - offset % page;
                const auto end = offset + destination.size();
                const auto remainder = end % page;
                const auto rounded_end = remainder == 0 || end > file_size - std::min(page - remainder, file_size)
                    ? file_size : end + (page - remainder);
                (void)::posix_fadvise(fd, static_cast<off_t>(begin),
                                      static_cast<off_t>(rounded_end - begin), POSIX_FADV_DONTNEED);
            }
        }
#endif
        bytes_read.fetch_add(destination.size(), std::memory_order_relaxed);
    }

    std::filesystem::path path;
    std::uint64_t file_size{};
    mutable std::atomic<std::uint64_t> bytes_read{0};
#if defined(_WIN32)
    mutable std::ifstream stream;
    mutable std::mutex mutex;
#else
    int fd{-1};
#endif
};

ModelFile::ModelFile(std::filesystem::path path)
    : impl_(std::make_unique<Impl>(std::move(path))) {}
ModelFile::~ModelFile() = default;
std::uint64_t ModelFile::size() const noexcept { return impl_ ? impl_->file_size : 0; }
std::uint64_t ModelFile::bytes_read() const noexcept {
    return impl_ ? impl_->bytes_read.load(std::memory_order_relaxed) : 0;
}
void ModelFile::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    if (!impl_) throw std::logic_error("model file is unavailable");
    impl_->read_into(offset, destination);
}
void ModelFile::read_into_uncached(std::uint64_t offset,
                                   std::span<std::byte> destination) const {
    if (!impl_) throw std::logic_error("model file is unavailable");
    impl_->read_into(offset, destination, true);
}
std::vector<std::byte> ModelFile::read(std::uint64_t offset, std::size_t size) const {
    std::vector<std::byte> result(size);
    read_into(offset, result);
    return result;
}

}  // namespace pokitlms::storage
