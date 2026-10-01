#include "pokitlms/storage/expert_store.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pokitlms::storage {

std::vector<ExpertSlice> split_expert_tensor(const model::TensorInfo& tensor,
                                             std::size_t expert_count) {
    if (expert_count == 0 || tensor.dimensions.empty() ||
        tensor.dimensions.back() != expert_count || !tensor.payload_size ||
        *tensor.payload_size % expert_count != 0 ||
        *tensor.payload_size > std::numeric_limits<std::uint64_t>::max() - tensor.file_offset) {
        throw std::invalid_argument("tensor cannot be split into expert slices: " + tensor.name);
    }
    const auto bytes_per_expert = *tensor.payload_size / expert_count;
    if (bytes_per_expert > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("expert slice exceeds addressable memory: " + tensor.name);
    }
    std::vector<ExpertSlice> slices;
    slices.reserve(expert_count);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        slices.push_back({tensor.file_offset + static_cast<std::uint64_t>(expert) * bytes_per_expert,
                          static_cast<std::size_t>(bytes_per_expert)});
    }
    return slices;
}

class ExpertStore::Impl {
public:
    Impl(std::shared_ptr<ModelFile> model_file, std::vector<ExpertSlice> index,
         std::size_t capacity)
        : file(std::move(model_file)), experts(std::move(index)), capacity_bytes(capacity) {
        if (!file) throw std::invalid_argument("expert store requires an open model file");
        for (const auto& expert : experts) {
            if (expert.offset > file->size() || expert.size > file->size() - expert.offset) {
                throw std::invalid_argument("expert slice extends beyond model file");
            }
        }
    }

    std::shared_ptr<ModelFile> file;
    std::vector<ExpertSlice> experts;
    std::size_t capacity_bytes;
    std::size_t resident_bytes{};
    std::uint64_t disk_bytes{};
    std::uint64_t hits{};
    std::uint64_t misses{};
    mutable std::mutex mutex;
    using Entry = std::pair<std::size_t, std::shared_ptr<const std::vector<std::byte>>>;
    std::list<Entry> lru;
    std::unordered_map<std::size_t, std::list<Entry>::iterator> lookup;
};

ExpertStore::ExpertStore(std::filesystem::path path, std::vector<ExpertSlice> experts,
                         std::size_t cache_capacity_bytes)
    : ExpertStore(std::make_shared<ModelFile>(std::move(path)), std::move(experts),
                  cache_capacity_bytes) {}

ExpertStore::ExpertStore(std::shared_ptr<ModelFile> file, std::vector<ExpertSlice> experts,
                         std::size_t cache_capacity_bytes)
    : impl_(std::make_unique<Impl>(std::move(file), std::move(experts), cache_capacity_bytes)) {}

ExpertStore::~ExpertStore() = default;
ExpertStore::ExpertStore(ExpertStore&&) noexcept = default;
ExpertStore& ExpertStore::operator=(ExpertStore&&) noexcept = default;

std::shared_ptr<const std::vector<std::byte>> ExpertStore::get(std::size_t expert_id) {
    if (!impl_) throw std::logic_error("expert store has been moved from");
    if (expert_id >= impl_->experts.size()) throw std::out_of_range("expert id out of range");

    {
        std::lock_guard lock(impl_->mutex);
        if (const auto found = impl_->lookup.find(expert_id); found != impl_->lookup.end()) {
            ++impl_->hits;
            impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second);
            return found->second->second;
        }
        ++impl_->misses;
    }

    const auto slice = impl_->experts[expert_id];
    auto bytes = std::make_shared<std::vector<std::byte>>(slice.size);
    impl_->file->read_into(slice.offset, *bytes);

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

std::uint64_t ExpertStore::cache_hits() const {
    if (!impl_) return 0;
    std::lock_guard lock(impl_->mutex);
    return impl_->hits;
}

std::uint64_t ExpertStore::cache_misses() const {
    if (!impl_) return 0;
    std::lock_guard lock(impl_->mutex);
    return impl_->misses;
}

void ExpertStore::clear_cache() {
    if (!impl_) return;
    std::lock_guard lock(impl_->mutex);
    impl_->lru.clear();
    impl_->lookup.clear();
    impl_->resident_bytes = 0;
}

}  // namespace pokitlms::storage
