#!/usr/bin/env python3
"""Generate compact Unicode L/M/N/White_Space ranges for Qwen's pre-tokenizer."""

import unicodedata
from pathlib import Path


def ranges_for(predicate):
    result = []
    start = previous = None
    for codepoint in range(0x110000):
        if predicate(codepoint):
            if start is None:
                start = codepoint
            previous = codepoint
        elif start is not None:
            result.append((start, previous))
            start = previous = None
    if start is not None:
        result.append((start, previous))
    return result


tables = {
    "kUnicodeLetters": ranges_for(lambda cp: unicodedata.category(chr(cp)).startswith("L")),
    "kUnicodeMarks": ranges_for(lambda cp: unicodedata.category(chr(cp)).startswith("M")),
    "kUnicodeNumbers": ranges_for(lambda cp: unicodedata.category(chr(cp)).startswith("N")),
    "kUnicodeWhitespace": ranges_for(
        lambda cp: unicodedata.category(chr(cp)).startswith("Z") or cp in (0x85, 0xA0)
        or 0x09 <= cp <= 0x0D
    ),
}

lines = [
    "// Generated from Python unicodedata " + unicodedata.unidata_version + ".",
    "// Unicode Character Database is distributed under the Unicode License.",
    "struct CodepointRange { char32_t first; char32_t last; };",
]
for name, ranges in tables.items():
    lines.append(f"static constexpr CodepointRange {name}[] = {{")
    lines.extend(f"    {{0x{first:X}, 0x{last:X}}}," for first, last in ranges)
    lines.append("};")

Path(__file__).resolve().parents[1].joinpath("src/unicode_ranges.inc").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
