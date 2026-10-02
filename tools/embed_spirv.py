#!/usr/bin/env python3
"""Embed build-time SPIR-V as uint32 arrays; no runtime shader compiler."""
import pathlib
import struct
import sys

target = pathlib.Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w") as out:
    out.write("#pragma once\n#include <cstdint>\nnamespace pokitlms::gpu::shaders {\n")
    for name, filename in zip(sys.argv[2::2], sys.argv[3::2]):
        payload = pathlib.Path(filename).read_bytes()
        words = struct.unpack("<" + "I" * (len(payload) // 4), payload)
        out.write(f"inline constexpr std::uint32_t {name}[] = {{\n")
        for i in range(0, len(words), 8):
            out.write("    " + ", ".join(f"0x{w:08x}U" for w in words[i:i+8]) + ",\n")
        out.write("};\n")
    out.write("}\n")
