#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pokitlms::model {

struct MetadataValue {
    using Array = std::vector<MetadataValue>;
    using Value = std::variant<std::uint64_t, std::int64_t, double, bool, std::string, Array>;
    Value value;
};

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type{};
    std::uint64_t file_offset{};
    std::optional<std::uint64_t> payload_size;
};

// Parses the GGUF v3 header and tensor directory. Tensor payloads remain on disk.
class GgufReader {
public:
    explicit GgufReader(std::filesystem::path path);

    [[nodiscard]] std::uint32_t version() const noexcept;
    [[nodiscard]] const std::map<std::string, MetadataValue, std::less<>>& metadata() const noexcept;
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const noexcept;
    [[nodiscard]] const TensorInfo* find_tensor(const std::string& name) const noexcept;
    [[nodiscard]] std::uint64_t data_offset() const noexcept;
    [[nodiscard]] std::uint64_t file_size() const noexcept;

private:
    std::uint32_t version_{};
    std::uint64_t data_offset_{};
    std::uint64_t file_size_{};
    std::map<std::string, MetadataValue, std::less<>> metadata_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, std::size_t> tensor_lookup_;
};

}  // namespace pokitlms::model
