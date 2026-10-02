#!/usr/bin/env python3
"""Generate the deterministic tiny GGUF model used by host and Android smoke tests."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return u64(len(encoded)) + encoded


def array(element_type: int, values: list[bytes]) -> bytes:
    return u32(9) + u32(element_type) + u64(len(values)) + b"".join(values)


def byte_symbols() -> list[str]:
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    next_codepoint = 256
    result = []
    for value in range(256):
        if value in visible:
            result.append(chr(value))
        else:
            result.append(chr(next_codepoint))
            next_codepoint += 1
    return result


def encode_gguf(
    metadata: list[tuple[str, bytes]],
    tensors: list[tuple[str, tuple[int, ...], list[float]]],
) -> bytes:
    encoded_metadata = b"".join(gguf_string(key) + value for key, value in metadata)

    def tensor_directory(offsets: list[int]) -> bytes:
        directory = bytearray()
        for (name, dimensions, _), offset in zip(tensors, offsets, strict=True):
            directory += gguf_string(name)
            directory += u32(len(dimensions))
            directory += b"".join(u64(dimension) for dimension in dimensions)
            directory += u32(0)  # GGML_TYPE_F32
            directory += u64(offset)
        return bytes(directory)

    empty_directory = tensor_directory([0] * len(tensors))
    header_size = 4 + 4 + 8 + 8 + len(encoded_metadata) + len(empty_directory)
    offsets: list[int] = []
    cursor = 0
    payloads: list[bytes] = []
    for _, _, values in tensors:
        cursor = (cursor + 31) & ~31
        offsets.append(cursor)
        payload = struct.pack(f"<{len(values)}f", *values)
        payloads.append(payload)
        cursor += len(payload)

    header = (
        b"GGUF"
        + u32(3)
        + u64(len(tensors))
        + u64(len(metadata))
        + encoded_metadata
        + tensor_directory(offsets)
    )
    data_start = (header_size + 31) & ~31
    if len(header) != header_size:
        raise AssertionError("GGUF header size changed while writing tensor offsets")
    result = bytearray(header)
    result += bytes(data_start - len(result))
    for offset, payload in zip(offsets, payloads, strict=True):
        if len(result) > data_start + offset:
            raise AssertionError("overlapping generated tensor payload")
        result += bytes(data_start + offset - len(result))
        result += payload
    return bytes(result)


def make_qwen35_model() -> bytes:
    symbols = byte_symbols()
    tokens = symbols + ["<|endoftext|>"]
    eos_id = len(tokens) - 1
    token_types = [1] * 256 + [3]

    metadata: list[tuple[str, bytes]] = []

    def string_metadata(key: str, value: str) -> None:
        metadata.append((key, u32(8) + gguf_string(value)))

    def integer_metadata(key: str, value: int) -> None:
        metadata.append((key, u32(4) + u32(value)))

    def float_metadata(key: str, value: float) -> None:
        metadata.append((key, u32(6) + struct.pack("<f", value)))

    def array_metadata(key: str, element_type: int, values: list[bytes]) -> None:
        metadata.append((key, array(element_type, values)))

    string_metadata("general.architecture", "qwen35")
    integer_metadata("general.alignment", 32)
    integer_metadata("qwen35.context_length", 512)
    integer_metadata("qwen35.embedding_length", 4)
    integer_metadata("qwen35.block_count", 2)
    integer_metadata("qwen35.feed_forward_length", 4)
    integer_metadata("qwen35.attention.head_count", 1)
    integer_metadata("qwen35.attention.head_count_kv", 1)
    integer_metadata("qwen35.attention.key_length", 2)
    integer_metadata("qwen35.attention.value_length", 2)
    integer_metadata("qwen35.rope.dimension_count", 2)
    integer_metadata("qwen35.ssm.conv_kernel", 2)
    integer_metadata("qwen35.ssm.state_size", 2)
    integer_metadata("qwen35.ssm.group_count", 1)
    integer_metadata("qwen35.ssm.time_step_rank", 1)
    integer_metadata("qwen35.ssm.inner_size", 2)
    integer_metadata("qwen35.full_attention_interval", 2)
    float_metadata("qwen35.attention.layer_norm_rms_epsilon", 1.0e-5)
    float_metadata("qwen35.rope.freq_base", 10000.0)
    metadata.append(("qwen35.tie_word_embeddings", u32(7) + b"\x00"))
    string_metadata("tokenizer.ggml.pre", "qwen35")
    string_metadata("tokenizer.ggml.model", "gpt2")
    array_metadata("tokenizer.ggml.tokens", 8, [gguf_string(token) for token in tokens])
    array_metadata("tokenizer.ggml.token_type", 5, [struct.pack("<i", kind) for kind in token_types])
    array_metadata("tokenizer.ggml.merges", 8, [])
    integer_metadata("tokenizer.ggml.bos_token_id", eos_id)
    integer_metadata("tokenizer.ggml.eos_token_id", eos_id)

    vocab_size = len(tokens)
    tensors: list[tuple[str, tuple[int, ...], list[float]]] = []

    def add(name: str, dimensions: tuple[int, ...], values: list[float]) -> None:
        expected = math.prod(dimensions)
        if len(values) != expected:
            raise ValueError(f"{name}: expected {expected} values, got {len(values)}")
        tensors.append((name, dimensions, values))

    add("token_embd.weight", (4, vocab_size), [1.0, 0.0, 0.0, 0.0] * vocab_size)
    add("output_norm.weight", (4,), [0.0] * 4)
    output_rows = [[0.0] * 4 for _ in range(vocab_size)]
    output_rows[ord("a")] = [1.0, 0.0, 0.0, 0.0]
    add("output.weight", (4, vocab_size), [value for row in output_rows for value in row])

    add("blk.0.attn_norm.weight", (4,), [0.0] * 4)
    add("blk.0.post_attention_norm.weight", (4,), [0.0] * 4)
    add("blk.0.ffn_gate.weight", (4, 4), [0.0] * 16)
    add("blk.0.ffn_up.weight", (4, 4), [0.0] * 16)
    add("blk.0.ffn_down.weight", (4, 4), [0.0] * 16)
    add("blk.0.attn_qkv.weight", (4, 6), [0.0] * 24)
    add("blk.0.attn_gate.weight", (4, 2), [0.0] * 8)
    add("blk.0.ssm_a", (1,), [0.0])
    add("blk.0.ssm_alpha.weight", (4, 1), [0.0] * 4)
    add("blk.0.ssm_beta.weight", (4, 1), [0.0] * 4)
    add("blk.0.ssm_conv1d.weight", (2, 6), [0.0] * 12)
    add("blk.0.ssm_dt.bias", (1,), [0.0])
    add("blk.0.ssm_norm.weight", (2,), [1.0] * 2)
    add("blk.0.ssm_out.weight", (2, 4), [0.0] * 8)

    add("blk.1.attn_norm.weight", (4,), [0.0] * 4)
    add("blk.1.post_attention_norm.weight", (4,), [0.0] * 4)
    add("blk.1.ffn_gate.weight", (4, 4), [0.0] * 16)
    add("blk.1.ffn_up.weight", (4, 4), [0.0] * 16)
    add("blk.1.ffn_down.weight", (4, 4), [0.0] * 16)
    add("blk.1.attn_q.weight", (4, 4), [0.0] * 16)
    add("blk.1.attn_k.weight", (4, 2), [0.0] * 8)
    add("blk.1.attn_v.weight", (4, 2), [0.0] * 8)
    add("blk.1.attn_q_norm.weight", (2,), [0.0] * 2)
    add("blk.1.attn_k_norm.weight", (2,), [0.0] * 2)
    add("blk.1.attn_output.weight", (2, 4), [0.0] * 8)

    return encode_gguf(metadata, tensors)


def make_qwen3moe_model() -> bytes:
    tokens = byte_symbols() + ["<|endoftext|>"]
    eos_id = len(tokens) - 1
    vocab_size = len(tokens)
    metadata: list[tuple[str, bytes]] = []

    def string_metadata(key: str, value: str) -> None:
        metadata.append((key, u32(8) + gguf_string(value)))

    def integer_metadata(key: str, value: int) -> None:
        metadata.append((key, u32(4) + u32(value)))

    def float_metadata(key: str, value: float) -> None:
        metadata.append((key, u32(6) + struct.pack("<f", value)))

    def array_metadata(key: str, element_type: int, values: list[bytes]) -> None:
        metadata.append((key, array(element_type, values)))

    string_metadata("general.architecture", "qwen3moe")
    integer_metadata("general.alignment", 32)
    integer_metadata("qwen3moe.context_length", 512)
    integer_metadata("qwen3moe.embedding_length", 4)
    integer_metadata("qwen3moe.block_count", 1)
    integer_metadata("qwen3moe.feed_forward_length", 4)
    integer_metadata("qwen3moe.attention.head_count", 1)
    integer_metadata("qwen3moe.attention.head_count_kv", 1)
    integer_metadata("qwen3moe.attention.key_length", 2)
    integer_metadata("qwen3moe.attention.value_length", 2)
    integer_metadata("qwen3moe.rope.dimension_count", 2)
    integer_metadata("qwen3moe.expert_count", 2)
    integer_metadata("qwen3moe.expert_used_count", 1)
    integer_metadata("qwen3moe.expert_feed_forward_length", 2)
    float_metadata("qwen3moe.attention.layer_norm_rms_epsilon", 1.0e-5)
    float_metadata("qwen3moe.rope.freq_base", 10000.0)
    metadata.append(("qwen3moe.tie_word_embeddings", u32(7) + b"\x00"))
    string_metadata("tokenizer.ggml.pre", "qwen2")
    string_metadata("tokenizer.ggml.model", "gpt2")
    array_metadata("tokenizer.ggml.tokens", 8, [gguf_string(token) for token in tokens])
    array_metadata("tokenizer.ggml.token_type", 5,
                   [struct.pack("<i", 1)] * 256 + [struct.pack("<i", 3)])
    array_metadata("tokenizer.ggml.merges", 8, [])
    integer_metadata("tokenizer.ggml.bos_token_id", eos_id)
    integer_metadata("tokenizer.ggml.eos_token_id", eos_id)

    tensors: list[tuple[str, tuple[int, ...], list[float]]] = []

    def add(name: str, dimensions: tuple[int, ...], values: list[float]) -> None:
        expected = math.prod(dimensions)
        if len(values) != expected:
            raise ValueError(f"{name}: expected {expected} values, got {len(values)}")
        tensors.append((name, dimensions, values))

    add("token_embd.weight", (4, vocab_size), [1.0, 0.0, 0.0, 0.0] * vocab_size)
    add("output_norm.weight", (4,), [1.0] * 4)
    output_rows = [[0.0] * 4 for _ in range(vocab_size)]
    output_rows[ord("a")] = [1.0, 0.0, 0.0, 0.0]
    add("output.weight", (4, vocab_size), [value for row in output_rows for value in row])

    add("blk.0.attn_norm.weight", (4,), [1.0] * 4)
    add("blk.0.attn_q.weight", (4, 2), [0.0] * 8)
    add("blk.0.attn_q_norm.weight", (2,), [1.0] * 2)
    add("blk.0.attn_k.weight", (4, 2), [0.0] * 8)
    add("blk.0.attn_k_norm.weight", (2,), [1.0] * 2)
    add("blk.0.attn_v.weight", (4, 2), [0.0] * 8)
    add("blk.0.attn_output.weight", (2, 4), [0.0] * 8)
    add("blk.0.ffn_norm.weight", (4,), [1.0] * 4)
    add("blk.0.ffn_gate_inp.weight", (4, 2), [0.0] * 8)
    add("blk.0.ffn_gate_exps.weight", (4, 2, 2), [0.0] * 16)
    add("blk.0.ffn_up_exps.weight", (4, 2, 2), [0.0] * 16)
    add("blk.0.ffn_down_exps.weight", (2, 4, 2), [0.0] * 16)
    return encode_gguf(metadata, tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--architecture", choices=("qwen35", "qwen3moe"), default="qwen35")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = make_qwen35_model() if args.architecture == "qwen35" else make_qwen3moe_model()
    args.output.write_bytes(model)


if __name__ == "__main__":
    main()
