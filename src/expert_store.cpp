#include "pokitlms/storage/expert_store.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pokitlms::storage {

class ExpertStore::Impl {
public:
    Impl(std::filesystem::path file, std::vector<ExpertSlice> index,
         std::size_t capacity)
        : path(std::move(file)), experts(std::move(index)), capacity_bytes(capacity),
          stream(path, std::ios::binary) {
        if (!stream) {
            throw std::runtime_error("cannot open model weights: " + path.string());
        }
        stream.seekg(0, std::ios::end);
        const auto end = stream.tellg();
        if (end < 0) {
            throw std::runtime_error("cannot determine model file size: " + path.string());
        }
        file_size = static_cast<std::uint64_t>(end);
        for (const auto& expert : experts) {
            if (expert.offset > file_size || expert.size > file_size - expert.offset) {
                throw std::invalid_argument("expert slice extends beyond model file");
            }
        }
    }

    std::filesystem::path path;
    std::vector<ExpertSlice> experts;
    std::size_t capacity_bytes;
    std::size_t resident_bytes{};
    std::uint64_t disk_bytes{};
    std::uint64_t file_size{};
    std::ifstream stream;
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
    std::lock_guard lock(impl_->mutex);
    if (expert_id >= impl_->experts.size()) throw std::out_of_range("expert id out of range");

    if (const auto found = impl_->lookup.find(expert_id); found != impl_->lookup.end()) {
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second);
        return found->second->second;
    }

    const auto slice = impl_->experts[expert_id];
    auto bytes = std::make_shared<std::vector<std::byte>>(slice.size);
    impl_->stream.clear();
    impl_->stream.seekg(static_cast<std::streamoff>(slice.offset), std::ios::beg);
    if (!impl_->stream || (slice.size != 0 &&
        !impl_->stream.read(reinterpret_cast<char*>(bytes->data()),
                            static_cast<std::streamsize>(slice.size)))) {
        throw std::runtime_error("failed to read expert " + std::to_string(expert_id) +
                                 " from " + impl_->path.string());
    }
    impl_->disk_bytes += slice.size;

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
