#pragma once

#include "pokitlms/model/gguf_reader.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pokitlms::model {

// Qwen's GPT-2 byte-level BPE tokenizer as stored in GGUF metadata.
class QwenBpeTokenizer {
public:
    explicit QwenBpeTokenizer(const GgufReader& model);
    ~QwenBpeTokenizer();
    QwenBpeTokenizer(QwenBpeTokenizer&&) noexcept;
    QwenBpeTokenizer& operator=(QwenBpeTokenizer&&) noexcept;
    QwenBpeTokenizer(const QwenBpeTokenizer&) = delete;
    QwenBpeTokenizer& operator=(const QwenBpeTokenizer&) = delete;

    [[nodiscard]] std::vector<std::uint32_t> encode(const std::string& utf8) const;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& tokens) const;
    [[nodiscard]] std::uint32_t bos_token_id() const noexcept;
    [[nodiscard]] std::uint32_t eos_token_id() const noexcept;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::model
