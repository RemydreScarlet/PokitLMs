#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace pokitlms::storage {

// Thread-safe positioned reads from a model file. POSIX builds use pread so
// parallel expert reads do not serialize on a shared seek position.
class ModelFile {
public:
    explicit ModelFile(std::filesystem::path path);
    ~ModelFile();
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read() const noexcept;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    // Reads bytes into the caller's buffer, then asks the OS to discard the
    // corresponding file-cache pages. The advice is best-effort and falls back
    // to a regular read on platforms without a page-cache discard API.
    void read_into_uncached(std::uint64_t offset, std::span<std::byte> destination) const;
    // Hint that a future read is likely. This does not alter returned weight
    // values and is a no-op on platforms without positioned-read advice.
    void prefetch(std::uint64_t offset, std::size_t size) const noexcept;
    [[nodiscard]] std::vector<std::byte> read(std::uint64_t offset, std::size_t size) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::storage
