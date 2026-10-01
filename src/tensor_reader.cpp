#include "pokitlms/model/tensor_reader.hpp"

#include <bit>
#include <cstring>
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

TensorReader::TensorReader(TensorInfo tensor,
                           std::shared_ptr<const std::vector<std::byte>> payload)
    : memory_(std::move(payload)), tensor_(std::move(tensor)) {
    if (!memory_ || tensor_.dimensions.empty() || !tensor_.payload_size ||
        *tensor_.payload_size != memory_->size()) {
        throw std::invalid_argument("in-memory tensor reader requires an exact payload");
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
    row_bytes_ = static_cast<std::size_t>(*tensor_.payload_size / row_count_);
}

const TensorInfo& TensorReader::tensor() const noexcept { return tensor_; }
std::uint64_t TensorReader::row_count() const noexcept { return row_count_; }
std::size_t TensorReader::row_bytes() const noexcept { return row_bytes_; }

void TensorReader::read_rows_into(std::uint64_t first_row, std::size_t rows,
                                  std::span<std::byte> destination) const {
    if (first_row > row_count_ || rows > row_count_ - first_row) {
        throw std::out_of_range("tensor row range is out of bounds");
    }
    if (rows != 0 && row_bytes_ > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("tensor row read is too large");
    }
    const auto byte_count = row_bytes_ * rows;
    if (destination.size() != byte_count) {
        throw std::invalid_argument("tensor row destination has the wrong size");
    }
    if (row_bytes_ != 0 && first_row > std::numeric_limits<std::uint64_t>::max() / row_bytes_) {
        throw std::out_of_range("tensor row offset overflows");
    }
    const auto byte_offset = first_row * static_cast<std::uint64_t>(row_bytes_);
    if (byte_offset > std::numeric_limits<std::uint64_t>::max() - tensor_.file_offset) {
        throw std::out_of_range("tensor row offset overflows");
    }
    if (memory_) {
        std::memcpy(destination.data(), memory_->data() + static_cast<std::size_t>(byte_offset),
                    destination.size());
        return;
    }
    file_->read_into(tensor_.file_offset + byte_offset, destination);
}

std::vector<std::byte> TensorReader::read_rows(std::uint64_t first_row,
                                              std::size_t rows) const {
    if (first_row > row_count_ || rows > row_count_ - first_row) {
        throw std::out_of_range("tensor row range is out of bounds");
    }
    if (rows != 0 && row_bytes_ > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("tensor row read is too large");
    }
    std::vector<std::byte> result(row_bytes_ * rows);
    read_rows_into(first_row, rows, result);
    return result;
}

std::vector<float> TensorReader::read_float_rows(std::uint64_t first_row,
                                                 std::size_t rows) const {
    if (tensor_.type != 0 && tensor_.type != 1 && tensor_.type != 30) {
        throw std::invalid_argument("float row access supports GGUF F32, F16, and BF16 tensors");
    }
    if (first_row > row_count_ || rows > row_count_ - first_row) {
        throw std::out_of_range("tensor row range is out of bounds");
    }
    if (rows != 0 && tensor_.dimensions.front() > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("float tensor row read is too large");
    }
    if (rows != 0 && row_bytes_ > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("tensor row read is too large");
    }
    std::vector<float> values(static_cast<std::size_t>(tensor_.dimensions.front()) * rows);
    std::vector<std::byte> encoded(row_bytes_ * rows);
    read_float_rows_into(first_row, rows, values, encoded);
    return values;
}

void TensorReader::read_float_rows_into(std::uint64_t first_row, std::size_t rows,
                                        std::span<float> values,
                                        std::span<std::byte> encoded) const {
    if (tensor_.type != 0 && tensor_.type != 1 && tensor_.type != 30) {
        throw std::invalid_argument("float row access supports GGUF F32, F16, and BF16 tensors");
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
    if (values.size() != static_cast<std::size_t>(elements_per_row) * rows) {
        throw std::invalid_argument("float tensor destination has the wrong size");
    }
    if (tensor_.type == 0 && std::endian::native == std::endian::little &&
        sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559) {
        read_rows_into(first_row, rows, std::as_writable_bytes(values));
        return;
    }
    read_rows_into(first_row, rows, encoded);
    if (tensor_.type == 0) {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::bit_cast<float>(read_u32(encoded.data() + i * 4));
    } else if (tensor_.type == 30) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = std::bit_cast<float>(static_cast<std::uint32_t>(read_u16(encoded.data() + i * 2)) << 16);
        }
    } else {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = half_to_float(read_u16(encoded.data() + i * 2));
    }
}

std::vector<std::byte> TensorReader::read_all() const {
    if (*tensor_.payload_size > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("tensor is too large to read into memory");
    }
    if (memory_) return *memory_;
    return file_->read(tensor_.file_offset, static_cast<std::size_t>(*tensor_.payload_size));
}

}  // namespace pokitlms::model
