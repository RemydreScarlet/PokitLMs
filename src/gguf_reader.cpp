#include "pokitlms/model/gguf_reader.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace pokitlms::model {
namespace {

constexpr std::uint64_t kMaxMetadataEntries = 1'000'000;
constexpr std::uint64_t kMaxTensors = 10'000'000;
constexpr std::uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr std::uint64_t kMaxArrayEntries = 100'000'000;
constexpr unsigned kMaxArrayDepth = 8;

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        if (!stream_) throw std::runtime_error("cannot open GGUF file: " + path.string());
        stream_.seekg(0, std::ios::end);
        const auto end = stream_.tellg();
        if (end < 0) throw std::runtime_error("cannot determine GGUF file size");
        size_ = static_cast<std::uint64_t>(end);
        stream_.seekg(0, std::ios::beg);
    }

    template <typename T>
    T integer() {
        static_assert(std::is_integral_v<T>);
        using U = std::make_unsigned_t<T>;
        U value = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            const auto byte = read_byte();
            value |= static_cast<U>(byte) << (i * 8);
        }
        return static_cast<T>(value);
    }

    std::string string() {
        const auto length = integer<std::uint64_t>();
        if (length > kMaxStringBytes || length > remaining()) {
            throw std::runtime_error("invalid or oversized GGUF string");
        }
        std::string result(static_cast<std::size_t>(length), '\0');
        if (length != 0 && !stream_.read(result.data(), static_cast<std::streamsize>(length))) {
            throw std::runtime_error("truncated GGUF string");
        }
        position_ += length;
        return result;
    }

    std::uint64_t position() const noexcept { return position_; }
    std::uint64_t size() const noexcept { return size_; }
    std::uint64_t remaining() const noexcept { return size_ - position_; }

private:
    std::uint8_t read_byte() {
        char byte{};
        if (!stream_.get(byte)) throw std::runtime_error("truncated GGUF file");
        ++position_;
        return static_cast<std::uint8_t>(static_cast<unsigned char>(byte));
    }
    std::ifstream stream_;
    std::uint64_t size_{};
    std::uint64_t position_{};
};

MetadataValue read_value(Reader& reader, std::uint32_t type, unsigned depth = 0) {
    switch (type) {
        case 0: return {{static_cast<std::uint64_t>(reader.integer<std::uint8_t>())}};
        case 1: return {{static_cast<std::int64_t>(reader.integer<std::int8_t>())}};
        case 2: return {{static_cast<std::uint64_t>(reader.integer<std::uint16_t>())}};
        case 3: return {{static_cast<std::int64_t>(reader.integer<std::int16_t>())}};
        case 4: return {{static_cast<std::uint64_t>(reader.integer<std::uint32_t>())}};
        case 5: return {{static_cast<std::int64_t>(reader.integer<std::int32_t>())}};
        case 6: return {{static_cast<double>(std::bit_cast<float>(reader.integer<std::uint32_t>()))}};
        case 7: {
            const auto value = reader.integer<std::uint8_t>();
            if (value > 1) throw std::runtime_error("invalid GGUF boolean");
            return {{value != 0}};
        }
        case 8: return {{reader.string()}};
        case 9: {
            if (depth >= kMaxArrayDepth) throw std::runtime_error("GGUF array nesting too deep");
            const auto element_type = reader.integer<std::uint32_t>();
            const auto count = reader.integer<std::uint64_t>();
            if (count > kMaxArrayEntries || count > reader.remaining()) {
                throw std::runtime_error("invalid or oversized GGUF array");
            }
            MetadataValue::Array array;
            array.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                array.push_back(read_value(reader, element_type, depth + 1));
            }
            return {{std::move(array)}};
        }
        case 10: return {{reader.integer<std::uint64_t>()}};
        case 11: return {{reader.integer<std::int64_t>()}};
        case 12: return {{std::bit_cast<double>(reader.integer<std::uint64_t>())}};
        default: throw std::runtime_error("unsupported GGUF metadata value type " + std::to_string(type));
    }
}

}  // namespace

GgufReader::GgufReader(std::filesystem::path path) {
    Reader reader(path);
    if (reader.integer<std::uint32_t>() != 0x46554747U) {
        throw std::runtime_error("invalid GGUF magic");
    }
    version_ = reader.integer<std::uint32_t>();
    if (version_ != 3) throw std::runtime_error("only GGUF version 3 is supported");
    const auto tensor_count = reader.integer<std::uint64_t>();
    const auto metadata_count = reader.integer<std::uint64_t>();
    if (tensor_count > kMaxTensors || metadata_count > kMaxMetadataEntries) {
        throw std::runtime_error("GGUF directory exceeds implementation limits");
    }

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        auto key = reader.string();
        const auto type = reader.integer<std::uint32_t>();
        auto value = read_value(reader, type);
        if (!metadata_.emplace(std::move(key), std::move(value)).second) {
            throw std::runtime_error("duplicate GGUF metadata key");
        }
    }

    tensors_.reserve(static_cast<std::size_t>(tensor_count));
    std::unordered_set<std::string> names;
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo tensor;
        tensor.name = reader.string();
        const auto dimensions = reader.integer<std::uint32_t>();
        if (dimensions == 0 || dimensions > 16) throw std::runtime_error("invalid GGUF tensor rank");
        tensor.dimensions.reserve(dimensions);
        for (std::uint32_t d = 0; d < dimensions; ++d) {
            const auto size = reader.integer<std::uint64_t>();
            if (size == 0) throw std::runtime_error("zero-sized GGUF tensor dimension");
            tensor.dimensions.push_back(size);
        }
        tensor.type = reader.integer<std::uint32_t>();
        tensor.file_offset = reader.integer<std::uint64_t>();
        if (!names.insert(tensor.name).second) throw std::runtime_error("duplicate GGUF tensor name");
        tensors_.push_back(std::move(tensor));
    }

    std::uint64_t alignment = 32;
    if (const auto it = metadata_.find("general.alignment"); it != metadata_.end()) {
        const auto* value = std::get_if<std::uint64_t>(&it->second.value);
        if (!value || *value == 0 || (*value & (*value - 1)) != 0) {
            throw std::runtime_error("invalid GGUF general.alignment");
        }
        alignment = *value;
    }
    const auto position = reader.position();
    const auto padding = (alignment - (position % alignment)) % alignment;
    if (padding > reader.remaining()) throw std::runtime_error("truncated GGUF alignment padding");
    data_offset_ = position + padding;
    file_size_ = reader.size();
    if (data_offset_ > file_size_) throw std::runtime_error("GGUF tensor data offset exceeds file");
    for (auto& tensor : tensors_) {
        if (tensor.file_offset % alignment != 0 || tensor.file_offset > file_size_ - data_offset_) {
            throw std::runtime_error("invalid GGUF tensor data offset");
        }
        tensor.file_offset += data_offset_;
    }
}

std::uint32_t GgufReader::version() const noexcept { return version_; }
const std::map<std::string, MetadataValue, std::less<>>& GgufReader::metadata() const noexcept { return metadata_; }
const std::vector<TensorInfo>& GgufReader::tensors() const noexcept { return tensors_; }
const TensorInfo* GgufReader::find_tensor(const std::string& name) const noexcept {
    for (const auto& tensor : tensors_) if (tensor.name == name) return &tensor;
    return nullptr;
}
std::uint64_t GgufReader::data_offset() const noexcept { return data_offset_; }
std::uint64_t GgufReader::file_size() const noexcept { return file_size_; }

}  // namespace pokitlms::model
