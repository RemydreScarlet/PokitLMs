#include "pokitlms/model/tensor_reader.hpp"

#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pokitlms::model {
namespace {

std::uint16_t read_u16(const std::byte* bytes) {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0])) |
           static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1]) << 8);
}

std::uint32_t read_u32(const std::byte* bytes) {
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[3])) << 24);
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fU;
    std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) return std::bit_cast<float>(sign);
        int unbiased = -14;
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            --unbiased;
        }
        mantissa &= 0x03ffU;
        result = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23) | (mantissa << 13);
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(result);
}

}  // namespace

TensorReader::TensorReader(std::shared_ptr<storage::ModelFile> file, TensorInfo tensor)
    : file_(std::move(file)), tensor_(std::move(tensor)) {
    if (!file_ || tensor_.dimensions.empty() || !tensor_.payload_size) {
        throw std::invalid_argument("tensor reader requires a file and known tensor extent");
    }
    row_count_ = 1;
    for (std::size_t i = 1; i < tensor_.dimensions.size(); ++i) {
        if (tensor_.dimensions[i] != 0 &&
            row_count_ > std::numeric_limits<std::uint64_t>::max() / tensor_.dimensions[i]) {
            throw std::invalid_argument("tensor row count overflows");
        }
        row_count_ *= tensor_.dimensions[i];
    }
    if (row_count_ == 0 || *tensor_.payload_size % row_count_ != 0) {
        throw std::invalid_argument("tensor payload does not divide into rows");
    }
    const auto bytes = *tensor_.payload_size / row_count_;
    if (bytes > std::numeric_limits<std::size_t>::max() ||
        tensor_.file_offset > file_->size() || *tensor_.payload_size > file_->size() - tensor_.file_offset) {
        throw std::invalid_argument("tensor payload exceeds file or address space");
    }
    row_bytes_ = static_cast<std::size_t>(bytes);
}

const TensorInfo& TensorReader::tensor() const noexcept { return tensor_; }
std::uint64_t TensorReader::row_count() const noexcept { return row_count_; }
std::size_t TensorReader::row_bytes() const noexcept { return row_bytes_; }

std::vector<std::byte> TensorReader::read_rows(std::uint64_t first_row,
                                              std::size_t rows) const {
    if (first_row > row_count_ || rows > row_count_ - first_row) {
        throw std::out_of_range("tensor row range is out of bounds");
    }
    if (rows != 0 && row_bytes_ > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("tensor row read is too large");
    }
    if (row_bytes_ != 0 && first_row > std::numeric_limits<std::uint64_t>::max() / row_bytes_) {
        throw std::out_of_range("tensor row offset overflows");
    }
    const auto byte_offset = first_row * static_cast<std::uint64_t>(row_bytes_);
    if (byte_offset > std::numeric_limits<std::uint64_t>::max() - tensor_.file_offset) {
        throw std::out_of_range("tensor row offset overflows");
    }
    return file_->read(tensor_.file_offset + byte_offset, row_bytes_ * rows);
}

std::vector<float> TensorReader::read_float_rows(std::uint64_t first_row,
                                                 std::size_t rows) const {
    if (tensor_.type != 0 && tensor_.type != 1) {
        throw std::invalid_argument("float row access only supports GGUF F32 and F16 tensors");
    }
    const std::size_t element_bytes = tensor_.type == 0 ? 4 : 2;
    const auto elements_per_row = tensor_.dimensions.front();
    if (row_bytes_ % element_bytes != 0 ||
        elements_per_row > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::invalid_argument("tensor row dimensions do not match floating point storage");
    }
    const auto expected_bytes = static_cast<std::size_t>(elements_per_row) * element_bytes;
    if (expected_bytes != row_bytes_) throw std::invalid_argument("invalid floating point row size");
    if (rows != 0 && elements_per_row > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("float tensor row read is too large");
    }
    const auto bytes = read_rows(first_row, rows);
    std::vector<float> values(static_cast<std::size_t>(elements_per_row) * rows);
    if (tensor_.type == 0) {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::bit_cast<float>(read_u32(bytes.data() + i * 4));
    } else {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = half_to_float(read_u16(bytes.data() + i * 2));
    }
    return values;
}

std::vector<std::byte> TensorReader::read_all() const {
    if (*tensor_.payload_size > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("tensor is too large to read into memory");
    }
    return file_->read(tensor_.file_offset, static_cast<std::size_t>(*tensor_.payload_size));
}

}  // namespace pokitlms::model
