#include "pokitlms/storage/expert_store.hpp"

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pokitlms::storage {

class ExpertStore::Impl {
public:
    Impl(std::filesystem::path file, std::vector<ExpertSlice> index,
         std::size_t capacity)
        : path(std::move(file)), experts(std::move(index)), capacity_bytes(capacity) {
#if defined(_WIN32)
        stream.open(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot open model weights: " + path.string());
        }
        stream.seekg(0, std::ios::end);
        const auto end = stream.tellg();
        if (end < 0) {
            throw std::runtime_error("cannot determine model file size: " + path.string());
        }
        file_size = static_cast<std::uint64_t>(end);
#else
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open model weights: " + path.string());
        struct stat info {};
        if (::fstat(fd, &info) != 0 || info.st_size < 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("cannot determine model file size: " + path.string());
        }
        file_size = static_cast<std::uint64_t>(info.st_size);
#endif
        for (const auto& expert : experts) {
            if (expert.offset > file_size || expert.size > file_size - expert.offset) {
#if !defined(_WIN32)
                ::close(fd);
                fd = -1;
#endif
                throw std::invalid_argument("expert slice extends beyond model file");
            }
        }
    }

    ~Impl() {
#if !defined(_WIN32)
        if (fd >= 0) ::close(fd);
#endif
    }

    void read_slice(std::uint64_t offset, std::vector<std::byte>& bytes) {
#if defined(_WIN32)
        std::lock_guard lock(stream_mutex);
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!stream || (!bytes.empty() && !stream.read(reinterpret_cast<char*>(bytes.data()),
                                                       static_cast<std::streamsize>(bytes.size())))) {
            throw std::runtime_error("failed to read model weight slice: " + path.string());
        }
#else
        std::size_t completed = 0;
        while (completed < bytes.size()) {
            const auto count = ::pread(fd, bytes.data() + completed, bytes.size() - completed,
                                       static_cast<off_t>(offset + completed));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("failed to read model weight slice: " + path.string());
            completed += static_cast<std::size_t>(count);
        }
#endif
    }

    std::filesystem::path path;
    std::vector<ExpertSlice> experts;
    std::size_t capacity_bytes;
    std::size_t resident_bytes{};
    std::uint64_t disk_bytes{};
    std::uint64_t file_size{};
#if defined(_WIN32)
    std::ifstream stream;
    std::mutex stream_mutex;
#else
    int fd{-1};
#endif
    mutable std::mutex mutex;
    using Entry = std::pair<std::size_t, std::shared_ptr<const std::vector<std::byte>>>;
    std::list<Entry> lru;
    std::unordered_map<std::size_t, std::list<Entry>::iterator> lookup;
};

ExpertStore::ExpertStore(std::filesystem::path path, std::vector<ExpertSlice> experts,
                         std::size_t cache_capacity_bytes)
    : impl_(std::make_unique<Impl>(std::move(path), std::move(experts),
                                   cache_capacity_bytes)) {}

ExpertStore::~ExpertStore() = default;
ExpertStore::ExpertStore(ExpertStore&&) noexcept = default;
ExpertStore& ExpertStore::operator=(ExpertStore&&) noexcept = default;

std::shared_ptr<const std::vector<std::byte>> ExpertStore::get(std::size_t expert_id) {
    if (!impl_) throw std::logic_error("expert store has been moved from");
    if (expert_id >= impl_->experts.size()) throw std::out_of_range("expert id out of range");

    {
        std::lock_guard lock(impl_->mutex);
        if (const auto found = impl_->lookup.find(expert_id); found != impl_->lookup.end()) {
            impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second);
            return found->second->second;
        }
    }

    const auto slice = impl_->experts[expert_id];
    if (slice.size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
        slice.offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("expert slice exceeds stream I/O limits");
    }
    auto bytes = std::make_shared<std::vector<std::byte>>(slice.size);
    impl_->read_slice(slice.offset, *bytes);

    std::lock_guard lock(impl_->mutex);
    impl_->disk_bytes += slice.size;
    if (const auto found = impl_->lookup.find(expert_id); found != impl_->lookup.end()) {
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second);
        return found->second->second;
    }
    if (slice.size <= impl_->capacity_bytes) {
        while (impl_->resident_bytes > impl_->capacity_bytes - slice.size &&
               !impl_->lru.empty()) {
            const auto last = std::prev(impl_->lru.end());
            impl_->resident_bytes -= last->second->size();
            impl_->lookup.erase(last->first);
            impl_->lru.erase(last);
        }
        impl_->lru.emplace_front(expert_id, bytes);
        impl_->lookup[expert_id] = impl_->lru.begin();
        impl_->resident_bytes += slice.size;
    }
    return bytes;
}

std::size_t ExpertStore::cache_capacity_bytes() const {
    return impl_ ? impl_->capacity_bytes : 0;
}

std::size_t ExpertStore::cache_bytes() const {
    if (!impl_) return 0;
    std::lock_guard lock(impl_->mutex);
    return impl_->resident_bytes;
}

std::uint64_t ExpertStore::bytes_read_from_disk() const {
    if (!impl_) return 0;
    std::lock_guard lock(impl_->mutex);
    return impl_->disk_bytes;
}

void ExpertStore::clear_cache() {
    if (!impl_) return;
    std::lock_guard lock(impl_->mutex);
    impl_->lru.clear();
    impl_->lookup.clear();
    impl_->resident_bytes = 0;
}

}  // namespace pokitlms::storage
