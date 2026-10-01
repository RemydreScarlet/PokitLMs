#include "pokitlms/model/qwen_bpe_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace pokitlms::model {
namespace {

#include "unicode_ranges.inc"

template <std::size_t N>
bool in_ranges(char32_t value, const CodepointRange (&ranges)[N]) {
    const auto* found = std::lower_bound(std::begin(ranges), std::end(ranges), value,
        [](const CodepointRange& range, char32_t cp) { return range.last < cp; });
    return found != std::end(ranges) && found->first <= value;
}

bool is_letter(char32_t cp) { return in_ranges(cp, kUnicodeLetters); }
bool is_number(char32_t cp) { return in_ranges(cp, kUnicodeNumbers); }
bool is_whitespace(char32_t cp) { return in_ranges(cp, kUnicodeWhitespace); }

struct Codepoint {
    char32_t value;
    std::size_t begin;
    std::size_t end;
};

std::vector<Codepoint> decode_utf8(std::string_view text) {
    std::vector<Codepoint> result;
    for (std::size_t i = 0; i < text.size();) {
        const auto begin = i;
        const auto lead = static_cast<std::uint8_t>(text[i]);
        char32_t cp;
        std::size_t count;
        if (lead < 0x80) { cp = lead; count = 1; }
        else if (lead >= 0xc2 && lead <= 0xdf) { cp = lead & 0x1f; count = 2; }
        else if (lead >= 0xe0 && lead <= 0xef) { cp = lead & 0x0f; count = 3; }
        else if (lead >= 0xf0 && lead <= 0xf4) { cp = lead & 0x07; count = 4; }
        else throw std::invalid_argument("tokenizer input is not valid UTF-8");
        if (i + count > text.size()) throw std::invalid_argument("truncated UTF-8 in tokenizer input");
        for (std::size_t j = 1; j < count; ++j) {
            const auto next = static_cast<std::uint8_t>(text[i + j]);
            if ((next & 0xc0) != 0x80) throw std::invalid_argument("invalid UTF-8 continuation byte");
            cp = (cp << 6) | (next & 0x3f);
        }
        if ((count == 2 && cp < 0x80) || (count == 3 && cp < 0x800) ||
            (count == 4 && cp < 0x10000) || cp > 0x10ffff ||
            (cp >= 0xd800 && cp <= 0xdfff)) {
            throw std::invalid_argument("non-canonical UTF-8 in tokenizer input");
        }
        i += count;
        result.push_back({cp, begin, i});
    }
    return result;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

const MetadataValue::Array& require_array(const GgufReader& model, const char* key) {
    const auto found = model.metadata().find(key);
    if (found == model.metadata().end()) throw std::runtime_error(std::string("missing GGUF metadata: ") + key);
    const auto* array = std::get_if<MetadataValue::Array>(&found->second.value);
    if (!array) throw std::runtime_error(std::string("GGUF metadata is not an array: ") + key);
    return *array;
}

std::uint64_t as_unsigned(const MetadataValue& value, const char* key) {
    if (const auto* u = std::get_if<std::uint64_t>(&value.value)) return *u;
    if (const auto* s = std::get_if<std::int64_t>(&value.value); s && *s >= 0) return static_cast<std::uint64_t>(*s);
    throw std::runtime_error(std::string("GGUF metadata is not an unsigned integer: ") + key);
}

std::uint64_t optional_unsigned(const GgufReader& model, const char* key, std::uint64_t fallback) {
    const auto found = model.metadata().find(key);
    return found == model.metadata().end() ? fallback : as_unsigned(found->second, key);
}

std::array<char32_t, 256> byte_to_unicode() {
    std::array<char32_t, 256> result{};
    std::array<bool, 256> assigned{};
    auto assign = [&](unsigned first, unsigned last) {
        for (unsigned byte = first; byte <= last; ++byte) {
            result[byte] = byte;
            assigned[byte] = true;
        }
    };
    assign(33, 126);
    assign(161, 172);
    assign(174, 255);
    char32_t extra = 256;
    for (unsigned byte = 0; byte < 256; ++byte) {
        if (!assigned[byte]) result[byte] = extra++;
    }
    return result;
}

std::unordered_map<char32_t, std::uint8_t> unicode_to_byte() {
    const auto forward = byte_to_unicode();
    std::unordered_map<char32_t, std::uint8_t> inverse;
    inverse.reserve(256);
    for (std::size_t i = 0; i < forward.size(); ++i) inverse.emplace(forward[i], static_cast<std::uint8_t>(i));
    return inverse;
}

std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

}  // namespace

struct QwenBpeTokenizer::Impl {
    struct TrieNode {
        std::unordered_map<unsigned char, std::size_t> next;
        std::uint32_t token_id{std::numeric_limits<std::uint32_t>::max()};
    };

    std::vector<std::string> vocab;
    std::vector<std::uint8_t> types;
    std::unordered_map<std::string, std::uint32_t> ids;
    std::unordered_map<std::uint64_t, std::uint32_t> merge_ranks;
    std::array<std::uint32_t, 256> byte_ids{};
    std::unordered_map<char32_t, std::uint8_t> byte_symbols;
    std::vector<TrieNode> special_trie{1};
    std::uint32_t bos{};
    std::uint32_t eos{};

    explicit Impl(const GgufReader& model) {
        const auto pre = model.metadata().find("tokenizer.ggml.pre");
        const auto* pre_name = pre == model.metadata().end() ? nullptr : std::get_if<std::string>(&pre->second.value);
        if (!pre_name || *pre_name != "qwen2") {
            throw std::runtime_error("Qwen tokenizer requires tokenizer.ggml.pre=qwen2");
        }
        const auto model_type = model.metadata().find("tokenizer.ggml.model");
        const auto* model_name = model_type == model.metadata().end() ? nullptr : std::get_if<std::string>(&model_type->second.value);
        if (!model_name || *model_name != "gpt2") {
            throw std::runtime_error("Qwen tokenizer requires tokenizer.ggml.model=gpt2");
        }

        const auto& token_values = require_array(model, "tokenizer.ggml.tokens");
        if (token_values.empty() || token_values.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("invalid Qwen tokenizer vocabulary size");
        }
        vocab.reserve(token_values.size());
        for (const auto& token : token_values) {
            const auto* text = std::get_if<std::string>(&token.value);
            if (!text) throw std::runtime_error("Qwen tokenizer vocabulary contains non-string entry");
            vocab.push_back(*text);
        }

        const auto& type_values = require_array(model, "tokenizer.ggml.token_type");
        if (type_values.size() != vocab.size()) throw std::runtime_error("Qwen token type count does not match vocabulary");
        types.reserve(type_values.size());
        for (const auto& type : type_values) {
            const auto value = as_unsigned(type, "tokenizer.ggml.token_type");
            if (value < 1 || value > 6) throw std::runtime_error("unknown Qwen token type");
            types.push_back(static_cast<std::uint8_t>(value));
        }

        ids.reserve(vocab.size());
        for (std::size_t id = 0; id < vocab.size(); ++id) {
            if (!ids.emplace(vocab[id], static_cast<std::uint32_t>(id)).second) {
                throw std::runtime_error("duplicate Qwen tokenizer token string");
            }
        }

        const auto symbols = byte_to_unicode();
        for (std::size_t byte = 0; byte < symbols.size(); ++byte) {
            std::string piece;
            append_utf8(piece, symbols[byte]);
            const auto found = ids.find(piece);
            if (found == ids.end() || types[found->second] != 1) {
                throw std::runtime_error("Qwen tokenizer is missing a base byte token");
            }
            byte_ids[byte] = found->second;
        }
        byte_symbols = unicode_to_byte();

        const auto& merges = require_array(model, "tokenizer.ggml.merges");
        if (merges.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Qwen merge table exceeds supported size");
        }
        merge_ranks.reserve(merges.size());
        for (std::size_t rank = 0; rank < merges.size(); ++rank) {
            const auto* line = std::get_if<std::string>(&merges[rank].value);
            if (!line) throw std::runtime_error("Qwen tokenizer merge is not a string");
            const auto separator = line->find(' ');
            if (separator == std::string::npos || separator == 0 || separator + 1 >= line->size()) {
                throw std::runtime_error("malformed Qwen tokenizer merge rule");
            }
            const auto left = ids.find(line->substr(0, separator));
            const auto right = ids.find(line->substr(separator + 1));
            if (left == ids.end() || right == ids.end()) throw std::runtime_error("Qwen merge references unknown token");
            merge_ranks.emplace(pair_key(left->second, right->second), static_cast<std::uint32_t>(rank));
        }

        for (std::size_t id = 0; id < vocab.size(); ++id) {
            if (types[id] == 2 || types[id] == 3 || types[id] == 4 || types[id] == 5) {
                std::size_t node = 0;
                for (const unsigned char byte : vocab[id]) {
                    const auto found = special_trie[node].next.find(byte);
                    if (found != special_trie[node].next.end()) {
                        node = found->second;
                    } else {
                        const auto child = special_trie.size();
                        special_trie[node].next.emplace(byte, child);
                        special_trie.emplace_back();
                        node = child;
                    }
                }
                special_trie[node].token_id = static_cast<std::uint32_t>(id);
            }
        }
        const auto bos_value = optional_unsigned(model, "tokenizer.ggml.bos_token_id", 0);
        const auto eos_value = optional_unsigned(model, "tokenizer.ggml.eos_token_id", 0);
        if (bos_value > std::numeric_limits<std::uint32_t>::max() ||
            eos_value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Qwen BOS/EOS token ID exceeds supported range");
        }
        bos = static_cast<std::uint32_t>(bos_value);
        eos = static_cast<std::uint32_t>(eos_value);
        if (bos >= vocab.size() || eos >= vocab.size()) throw std::runtime_error("Qwen BOS/EOS token ID out of range");
    }

    std::vector<std::uint32_t> bpe(std::string_view text) const {
        struct Node {
            std::uint32_t id{};
            std::size_t previous{};
            std::size_t next{};
            bool active{true};
        };
        struct Candidate { std::uint32_t rank; std::size_t left; std::size_t right; };
        struct Later {
            bool operator()(const Candidate& a, const Candidate& b) const noexcept {
                return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
            }
        };
        const auto none = std::numeric_limits<std::size_t>::max();
        std::vector<Node> nodes;
        nodes.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            nodes.push_back({byte_ids[static_cast<unsigned char>(text[i])],
                i == 0 ? none : i - 1, i + 1 == text.size() ? none : i + 1, true});
        }
        std::priority_queue<Candidate, std::vector<Candidate>, Later> candidates;
        auto enqueue = [&](std::size_t left) {
            if (left == none || !nodes[left].active || nodes[left].next == none) return;
            const auto right = nodes[left].next;
            const auto found = merge_ranks.find(pair_key(nodes[left].id, nodes[right].id));
            if (found != merge_ranks.end()) candidates.push({found->second, left, right});
        };
        for (std::size_t i = 0; i + 1 < nodes.size(); ++i) enqueue(i);
        while (!candidates.empty()) {
            const auto candidate = candidates.top();
            candidates.pop();
            auto& left = nodes[candidate.left];
            auto& right = nodes[candidate.right];
            if (!left.active || !right.active || left.next != candidate.right) continue;
            const auto current_rank = merge_ranks.find(pair_key(left.id, right.id));
            if (current_rank == merge_ranks.end() || current_rank->second != candidate.rank) continue;
            const auto combined = vocab[left.id] + vocab[right.id];
            const auto merged = ids.find(combined);
            if (merged == ids.end()) throw std::runtime_error("Qwen BPE merge result is absent from vocabulary");
            const auto before = left.previous;
            const auto after = right.next;
            left.id = merged->second;
            left.next = after;
            right.active = false;
            if (after != none) nodes[after].previous = candidate.left;
            enqueue(before);
            enqueue(candidate.left);
        }
        std::vector<std::uint32_t> result;
        result.reserve(nodes.size());
        for (const auto& node : nodes) if (node.active) result.push_back(node.id);
        return result;
    }

    std::size_t match_special(std::string_view text, std::size_t start, std::uint32_t& id) const {
        std::size_t node = 0;
        std::size_t best_end = start;
        id = std::numeric_limits<std::uint32_t>::max();
        for (std::size_t i = start; i < text.size(); ++i) {
            const auto next = special_trie[node].next.find(static_cast<unsigned char>(text[i]));
            if (next == special_trie[node].next.end()) break;
            node = next->second;
            if (special_trie[node].token_id != std::numeric_limits<std::uint32_t>::max()) {
                id = special_trie[node].token_id;
                best_end = i + 1;
            }
        }
        return best_end;
    }

    std::size_t pretoken_end(const std::vector<Codepoint>& cp, std::size_t start) const {
        const auto count = cp.size();
        auto value = [&](std::size_t i) { return cp[i].value; };
        auto contraction = [&](std::string_view suffix) {
            if (start + suffix.size() + 1 > count || value(start) != '\'') return false;
            for (std::size_t i = 0; i < suffix.size(); ++i) {
                if (value(start + i + 1) != static_cast<unsigned char>(suffix[i])) return false;
            }
            return true;
        };
        for (const auto suffix : {std::string_view("re"), std::string_view("ve"), std::string_view("ll"),
                                  std::string_view("s"), std::string_view("t"), std::string_view("m"),
                                  std::string_view("d")}) {
            if (contraction(suffix)) return start + suffix.size() + 1;
        }

        std::size_t content = start;
        if (value(start) == ' ' && start + 1 < count) content++;
        if (content < count && is_letter(value(content))) {
            std::size_t end = content + 1;
            while (end < count && is_letter(value(end))) ++end;
            return end;
        }
        if (content < count && is_number(value(content))) {
            std::size_t end = content + 1;
            while (end < count && is_number(value(end))) ++end;
            return end;
        }
        if (content < count && !is_whitespace(value(content)) &&
            !is_letter(value(content)) && !is_number(value(content))) {
            std::size_t end = content + 1;
            while (end < count && !is_whitespace(value(end)) &&
                   !is_letter(value(end)) && !is_number(value(end))) ++end;
            while (end < count && (value(end) == '\r' || value(end) == '\n')) ++end;
            return end;
        }

        if (is_whitespace(value(start))) {
            std::size_t run_end = start;
            std::size_t last_newline = count;
            while (run_end < count && is_whitespace(value(run_end))) {
                if (value(run_end) == '\r' || value(run_end) == '\n') last_newline = run_end;
                ++run_end;
            }
            if (last_newline != count) {
                std::size_t end = last_newline + 1;
                if (value(last_newline) == '\r' && end < count && value(end) == '\n') ++end;
                return end;
            }
            if (run_end == count) return run_end;
            return run_end - start > 1 ? run_end - 1 : run_end;
        }
        return start + 1;
    }

    void encode_plain(std::string_view text, std::vector<std::uint32_t>& output) const {
        const auto codepoints = decode_utf8(text);
        for (std::size_t start = 0; start < codepoints.size();) {
            const auto end = pretoken_end(codepoints, start);
            const auto byte_start = codepoints[start].begin;
            const auto byte_end = codepoints[end - 1].end;
            auto pieces = bpe(text.substr(byte_start, byte_end - byte_start));
            output.insert(output.end(), pieces.begin(), pieces.end());
            start = end;
        }
    }
};

QwenBpeTokenizer::QwenBpeTokenizer(const GgufReader& model) : impl_(std::make_unique<Impl>(model)) {}
QwenBpeTokenizer::~QwenBpeTokenizer() = default;
QwenBpeTokenizer::QwenBpeTokenizer(QwenBpeTokenizer&&) noexcept = default;
QwenBpeTokenizer& QwenBpeTokenizer::operator=(QwenBpeTokenizer&&) noexcept = default;

std::vector<std::uint32_t> QwenBpeTokenizer::encode(const std::string& utf8) const {
    if (!impl_) throw std::logic_error("tokenizer has been moved from");
    (void)decode_utf8(utf8);
    std::vector<std::uint32_t> output;
    std::size_t cursor = 0;
    std::size_t plain_start = 0;
    while (cursor < utf8.size()) {
        std::uint32_t special_id;
        const auto special_end = impl_->match_special(utf8, cursor, special_id);
        if (special_id == std::numeric_limits<std::uint32_t>::max()) {
            ++cursor;
            continue;
        }
        impl_->encode_plain(std::string_view(utf8).substr(plain_start, cursor - plain_start), output);
        output.push_back(special_id);
        cursor = special_end;
        plain_start = cursor;
    }
    impl_->encode_plain(std::string_view(utf8).substr(plain_start), output);
    return output;
}

std::string QwenBpeTokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
    if (!impl_) throw std::logic_error("tokenizer has been moved from");
    std::string output;
    for (const auto id : tokens) {
        if (id >= impl_->vocab.size()) throw std::out_of_range("token ID out of range");
        if (impl_->types[id] == 6) {
            const auto& token = impl_->vocab[id];
            if (token.size() != 6 || token.substr(0, 3) != "<0x" || token.back() != '>') {
                throw std::runtime_error("malformed Qwen byte token");
            }
            unsigned value = 0;
            for (std::size_t i = 3; i < 5; ++i) {
                const char c = token[i];
                value = value * 16 + (c >= '0' && c <= '9' ? c - '0' :
                    c >= 'A' && c <= 'F' ? c - 'A' + 10 : c >= 'a' && c <= 'f' ? c - 'a' + 10 : 255);
                if (value > 255) throw std::runtime_error("malformed Qwen byte token");
            }
            output.push_back(static_cast<char>(value));
        } else if (impl_->types[id] != 1) {
            output += impl_->vocab[id];
        } else {
            const auto chars = decode_utf8(impl_->vocab[id]);
            for (const auto& cp : chars) {
                const auto found = impl_->byte_symbols.find(cp.value);
                if (found == impl_->byte_symbols.end()) throw std::runtime_error("invalid byte-level BPE symbol");
                output.push_back(static_cast<char>(found->second));
            }
        }
    }
    return output;
}

std::uint32_t QwenBpeTokenizer::bos_token_id() const noexcept { return impl_ ? impl_->bos : 0; }
std::uint32_t QwenBpeTokenizer::eos_token_id() const noexcept { return impl_ ? impl_->eos : 0; }
std::size_t QwenBpeTokenizer::vocabulary_size() const noexcept { return impl_ ? impl_->vocab.size() : 0; }

}  // namespace pokitlms::model
