#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/model/qwen_bpe_tokenizer.hpp"
#include "pokitlms/ops/quantized_linear.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void u16(std::span<std::byte> bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::byte>(value & 255U);
    bytes[offset + 1] = static_cast<std::byte>(value >> 8U);
}

using Linear = void (*)(std::span<const float>, std::span<const std::byte>,
                        std::size_t, std::span<const float>, std::span<float>);

void check_block(std::uint32_t type, std::span<const std::byte> block,
                 const std::array<float, 256>& expected, Linear linear) {
    std::array<float, 256> decoded{};
    pokitlms::dequantize_quantized_row(type, block, decoded);
    std::array<float, 256> input{};
    float dot = 0.75F;
    for (std::size_t i = 0; i < input.size(); ++i) {
        require(decoded[i] == expected[i], "GGUF type " + std::to_string(type) +
            " decoded incorrectly at weight " + std::to_string(i));
        input[i] = static_cast<float>(static_cast<int>((i * 7) % 13) - 6) / 8.0F;
        dot += input[i] * expected[i];
    }
    std::array<float, 1> output{}, bias{0.75F};
    linear(input, block, 1, bias, output);
    require(std::fabs(output[0] - dot) < 0.01F, "quantized projection differs from stored weights");
}

void test_q3_k() {
    // GGUF layout: hmask[32], qs[64], scales[12], fp16 d.
    std::array<std::byte, 110> block{};
    std::array<float, 256> expected{};
    u16(block, 108, 0x3c00); // d = 1; signed scale -32, signed quant -4.
    expected.fill(128.0F);
    check_block(11, block, expected, pokitlms::linear_q3_k);

    block.fill(std::byte{0});
    u16(block, 108, 0x3800); // d = 0.5.
    for (std::size_t group = 0; group < 16; ++group) {
        const unsigned scale = static_cast<unsigned>((group * 7 + 11) % 64);
        block[96 + group % 8] |= static_cast<std::byte>((scale & 15U) << (4 * (group / 8)));
        block[104 + group % 4] |= static_cast<std::byte>((scale >> 4) << (2 * (group / 4)));
        for (std::size_t j = 0; j < 16; ++j) {
            const auto index = group * 16 + j;
            const int quant = static_cast<int>((index * 5 + group) % 8) - 4;
            const auto lane = index % 32;
            if (quant >= 0) block[lane] |= static_cast<std::byte>(1U << (index / 32));
            block[32 + (index / 128) * 32 + lane] |=
                static_cast<std::byte>((static_cast<unsigned>(quant) & 3U) << (2 * ((index % 128) / 32)));
            expected[index] = 0.5F * static_cast<float>(static_cast<int>(scale) - 32) * quant;
        }
    }
    check_block(11, block, expected, pokitlms::linear_q3_k);
}

void pack_scale_min(std::span<std::byte> block, std::size_t group, unsigned scale, unsigned minimum) {
    if (group < 4) {
        block[4 + group] |= static_cast<std::byte>(scale);
        block[8 + group] |= static_cast<std::byte>(minimum);
    } else {
        block[group] |= static_cast<std::byte>((scale >> 4) << 6);
        block[4 + group] |= static_cast<std::byte>((minimum >> 4) << 6);
        block[8 + group] |= static_cast<std::byte>((scale & 15U) | ((minimum & 15U) << 4));
    }
}

void test_q5_k() {
    // GGUF layout: fp16 d/dmin, scales[12], qh[32], qs[128].
    std::array<std::byte, 176> block{};
    std::array<float, 256> expected{};
    u16(block, 0, 0x3c00);
    for (std::size_t group = 0; group < 8; ++group) pack_scale_min(block, group, 1, 0);
    std::fill(block.begin() + 48, block.end(), std::byte{0x11});
    expected.fill(1.0F);
    check_block(13, block, expected, pokitlms::linear_q5_k);

    block.fill(std::byte{0});
    u16(block, 0, 0x3800); // d = 0.5.
    u16(block, 2, 0x3400); // dmin = 0.25.
    for (std::size_t group = 0; group < 8; ++group) {
        const unsigned scale = static_cast<unsigned>((group * 9 + 5) % 64);
        const unsigned minimum = static_cast<unsigned>((group * 11 + 7) % 64);
        pack_scale_min(block, group, scale, minimum);
        for (std::size_t lane = 0; lane < 32; ++lane) {
            const auto index = group * 32 + lane;
            const unsigned quant = static_cast<unsigned>((index * 13 + group) % 32);
            block[16 + lane] |= static_cast<std::byte>((quant >> 4) << group);
            block[48 + (group / 2) * 32 + lane] |= static_cast<std::byte>((quant & 15U) << (4 * (group % 2)));
            expected[index] = 0.5F * scale * quant - 0.25F * minimum;
        }
    }
    check_block(13, block, expected, pokitlms::linear_q5_k);
}

struct TokenCase {
    std::string input;
    std::vector<std::string> qwen2;
    std::vector<std::string> qwen35;
};

template <typename T>
void integer(std::ostream& out, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) out.put(static_cast<char>((value >> (8 * i)) & 255U));
}

void string(std::ostream& out, const std::string& value) {
    integer<std::uint64_t>(out, value.size());
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

class TokenFixture {
public:
    TokenFixture(const std::string& pre, const std::vector<TokenCase>& cases) {
        // A vocabulary with all substring merges exposes pre-token boundaries:
        // every piece becomes one token, while merges across pieces must be forbidden.
        std::array<std::string, 256> symbols;
        unsigned extra = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const auto cp = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174
                ? byte : extra++;
            if (cp < 128) symbols[byte].push_back(static_cast<char>(cp));
            else {
                symbols[byte].push_back(static_cast<char>(0xc0U | (cp >> 6)));
                symbols[byte].push_back(static_cast<char>(0x80U | (cp & 63U)));
            }
        }
        auto encoded = [&](const std::string& raw) {
            std::string result;
            for (unsigned char byte : raw) result += symbols[byte];
            return result;
        };
        std::vector<std::string> vocab;
        for (unsigned byte = 0; byte < 256; ++byte) {
            ids.emplace(std::string(1, static_cast<char>(byte)), byte);
            vocab.push_back(symbols[byte]);
        }
        std::set<std::string> pieces;
        for (const auto& item : cases) {
            for (std::size_t begin = 0; begin < item.input.size(); ++begin) {
                for (std::size_t size = 2; size <= item.input.size() - begin; ++size) {
                    pieces.insert(item.input.substr(begin, size));
                }
            }
        }
        std::vector<std::string> merges;
        for (const auto& raw : pieces) {
            ids.emplace(raw, static_cast<std::uint32_t>(vocab.size()));
            vocab.push_back(encoded(raw));
            for (std::size_t split = 1; split < raw.size(); ++split) {
                merges.push_back(encoded(raw.substr(0, split)) + " " + encoded(raw.substr(split)));
            }
        }
        path = std::filesystem::temp_directory_path() / ("pokit-token-regression-" + pre + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        std::ofstream out(path, std::ios::binary);
        require(static_cast<bool>(out), "cannot create tokenizer GGUF fixture");
        integer<std::uint32_t>(out, 0x46554747U);
        integer<std::uint32_t>(out, 3);
        integer<std::uint64_t>(out, 0); // No tensor payloads are needed for tokenization.
        integer<std::uint64_t>(out, 7);
        auto text_metadata = [&](const char* key, const std::string& value) {
            string(out, key); integer<std::uint32_t>(out, 8); string(out, value);
        };
        auto strings = [&](const char* key, const std::vector<std::string>& values) {
            string(out, key); integer<std::uint32_t>(out, 9); integer<std::uint32_t>(out, 8);
            integer<std::uint64_t>(out, values.size());
            for (const auto& value : values) string(out, value);
        };
        text_metadata("tokenizer.ggml.pre", pre);
        text_metadata("tokenizer.ggml.model", "gpt2");
        strings("tokenizer.ggml.tokens", vocab);
        string(out, "tokenizer.ggml.token_type");
        integer<std::uint32_t>(out, 9); integer<std::uint32_t>(out, 4);
        integer<std::uint64_t>(out, vocab.size());
        for (std::size_t i = 0; i < vocab.size(); ++i) integer<std::uint32_t>(out, 1);
        strings("tokenizer.ggml.merges", merges);
        for (const auto* key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"}) {
            string(out, key); integer<std::uint32_t>(out, 4); integer<std::uint32_t>(out, 0);
        }
        while (out.tellp() % 32 != 0) out.put(0);
        require(static_cast<bool>(out), "cannot write tokenizer GGUF fixture");
    }
    ~TokenFixture() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    std::filesystem::path path;
    std::unordered_map<std::string, std::uint32_t> ids;
};

void test_tokenizer() {
    const std::vector<TokenCase> cases{
        {"'Hello", {"'Hello"}, {"'Hello"}},
        {"’Hello", {"’Hello"}, {"’Hello"}},
        {"we'REady", {"we", "'RE", "ady"}, {"we", "'RE", "ady"}},
        {"I'm I'VE I'LL I'D", {"I", "'m", " I", "'VE", " I", "'LL", " I", "'D"},
                            {"I", "'m", " I", "'VE", " I", "'LL", " I", "'D"}},
        {" 12345", {" ", "1", "2", "3", "4", "5"}, {" ", "1", "2", "3", "4", "5"}},
        {"\tHello", {"\tHello"}, {"\tHello"}},
        {"—Hello", {"—Hello"}, {"—Hello"}},
        {"日本語123", {"日本語", "1", "2", "3"}, {"日本語", "1", "2", "3"}},
        {" \r\n  Hello", {" \r\n", " ", " Hello"}, {" \r\n", " ", " Hello"}},
        {" !!\r\n", {" !!\r\n"}, {" !!\r\n"}},
        {"áb", {"a", "́b"}, {"áb"}},
        {"á", {"a", "́"}, {"á"}},
        {"́̂Z", {"́̂", "Z"}, {"́̂Z"}},
        {"hello\tworld", {"hello", "\tworld"}, {"hello", "\tworld"}},
        {"text   ", {"text", "   "}, {"text", "   "}},
        {"1a²b", {"1", "a", "²", "b"}, {"1", "a", "²", "b"}},
    };
    for (const auto& pre : {std::string("qwen2"), std::string("qwen35")}) {
        TokenFixture fixture(pre, cases);
        pokitlms::model::GgufReader gguf(fixture.path);
        pokitlms::model::QwenBpeTokenizer tokenizer(gguf);
        for (const auto& item : cases) {
            std::vector<std::uint32_t> expected;
            const auto& pieces = pre == "qwen2" ? item.qwen2 : item.qwen35;
            for (const auto& piece : pieces) expected.push_back(fixture.ids.at(piece));
            const auto actual = tokenizer.encode(item.input);
            require(actual == expected, pre + " pre-token boundaries differ for: " + item.input);
            require(tokenizer.decode(actual) == item.input, "tokenizer does not preserve input bytes");
        }
    }
}
} // namespace

int main() {
    try {
        test_q3_k();
        test_q5_k();
        test_tokenizer();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
