#pragma once

#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/storage/model_file.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace pokitlms::model {

// Row-oriented access to one GGUF tensor. Rows follow the tensor's contiguous
// dimension-0 layout; large tensors can be consumed without reading all rows.
class TensorReader {
public:
    TensorReader(std::shared_ptr<storage::ModelFile> file, TensorInfo tensor);
    TensorReader(TensorInfo tensor, std::shared_ptr<const std::vector<std::byte>> payload);

    [[nodiscard]] const TensorInfo& tensor() const noexcept;
    [[nodiscard]] std::uint64_t row_count() const noexcept;
    [[nodiscard]] std::size_t row_bytes() const noexcept;
    void read_rows_into(std::uint64_t first_row, std::size_t row_count,
                        std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read_rows(std::uint64_t first_row,
                                                   std::size_t row_count) const;
    void read_float_rows_into(std::uint64_t first_row, std::size_t row_count,
                              std::span<float> destination,
                              std::span<std::byte> encoded_scratch) const;
    [[nodiscard]] std::vector<float> read_float_rows(std::uint64_t first_row,
                                                     std::size_t row_count) const;
    [[nodiscard]] std::vector<std::byte> read_all() const;

private:
    std::shared_ptr<storage::ModelFile> file_;
    std::shared_ptr<const std::vector<std::byte>> memory_;
    TensorInfo tensor_;
    std::uint64_t row_count_{};
    std::size_t row_bytes_{};
};

}  // namespace pokitlms::model
