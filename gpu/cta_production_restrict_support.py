"""Shared source and allocation guards for production ``__restrict__`` trials.

The helpers in this file do not select a decoder or change CUDA operations.
They make two proof obligations mechanical:

* only explicitly listed pointer parameters receive a qualifier; and
* every live top-level pointer argument names a disjoint device allocation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


# Kernel inputs use two names where the CUDA ABI intentionally differs from
# the HostModel/device dictionary.  Keeping every binding explicit prevents a
# newly added kernel pointer from silently escaping the allocation audit.
DEVICE_POINTER_BINDINGS: dict[str, str] = {
    "d2e_offsets": "d2e_offsets",
    "d2e": "d2e",
    "edet_offsets": "edet_offsets",
    "edets": "edets",
    "eneighbor_word_offsets": "eneighbor_word_offsets",
    "eneighbor_words": "eneighbor_words",
    "eneighbor_masks": "eneighbor_masks",
    "weights": "weights",
    "error_to_dem": "error_to_dem",
    "detector_order": "detector_order",
    "error_hash1": "error_hash1",
    "error_hash2": "error_hash2",
    "count_values": "count_values",
    "d2e_ranked": "d2e_ranked32",
    "count_ranks16": "count_ranks16",
    "shot_offsets": "shot_offsets",
    "shot_detections": "shot_detections",
    "initial_shot_bits": "shot_bits",
    "shot_hash1": "shot_hash1",
    "shot_hash2": "shot_hash2",
}


def _function_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        r"(?P<head>(?:extern \"C\" )?(?:__device__|__global__)"
        r"[^{};]*?\b" + re.escape(name) + r"\s*\()"
        r"(?P<params>[^)]*)(?P<tail>\)\s*\{)",
        re.DOTALL,
    )


def qualify_signature_parameters(
    source: str,
    contracts: Mapping[str, tuple[str, ...]],
    expected_definitions: Mapping[str, int] | None = None,
) -> str:
    """Qualify an explicit pointer allowlist with count-checked rewrites."""
    definitions = expected_definitions or {}
    transformed = source
    for function, names in contracts.items():
        pattern = _function_pattern(function)
        matches = list(pattern.finditer(transformed))
        expected = int(definitions.get(function, 1))
        if len(matches) != expected:
            raise RuntimeError(
                f"restrict audit expected {expected} definitions of {function}, "
                f"found {len(matches)}"
            )

        def qualify(match: re.Match[str]) -> str:
            params = match.group("params")
            for name in names:
                declaration = re.compile(
                    r"(?P<star>\*\s*)(?!__restrict__\s+)"
                    + re.escape(name)
                    + r"\b"
                )
                params, count = declaration.subn(
                    lambda item: item.group("star") + "__restrict__ " + name,
                    params,
                    count=1,
                )
                if count != 1:
                    raise RuntimeError(
                        f"{function}.{name} is missing, duplicated, or not a pointer"
                    )
            return match.group("head") + params + match.group("tail")

        transformed = pattern.sub(qualify, transformed)

    expected_qualifiers = sum(
        len(names) * int(definitions.get(function, 1))
        for function, names in contracts.items()
    )
    actual_qualifiers = (
        transformed.count("__restrict__") - source.count("__restrict__")
    )
    if actual_qualifiers != expected_qualifiers:
        raise RuntimeError(
            "unexpected restrict qualifier count: "
            f"expected {expected_qualifiers}, found {actual_qualifiers}"
        )
    return transformed


@dataclass(frozen=True)
class AllocationSpan:
    name: str
    begin: int
    end: int


def _array_span(name: str, array: Any) -> AllocationSpan | None:
    nbytes = int(array.nbytes)
    if nbytes == 0:
        return None
    begin = int(array.data.ptr)
    if begin == 0:
        raise ValueError(f"non-empty kernel buffer {name} has a null pointer")
    return AllocationSpan(name=name, begin=begin, end=begin + nbytes)


def assert_nonoverlapping_allocations(
    buffers: Mapping[str, Any],
) -> tuple[AllocationSpan, ...]:
    """Return sorted spans after proving all live byte ranges are disjoint."""
    spans = tuple(sorted(
        (
            span
            for name, array in buffers.items()
            if (span := _array_span(name, array)) is not None
        ),
        key=lambda span: (span.begin, span.end, span.name),
    ))
    for left, right in zip(spans, spans[1:]):
        if right.begin < left.end:
            raise ValueError(
                "__restrict__ allocation overlap: "
                f"{left.name}=[0x{left.begin:x},0x{left.end:x}) and "
                f"{right.name}=[0x{right.begin:x},0x{right.end:x})"
            )
    return spans


def audit_kernel_allocations(
    device: Mapping[str, Any],
    workspace: Mapping[str, Any],
    pointer_parameters: tuple[str, ...],
) -> dict[str, int]:
    """Check every restricted top-level pointer against the live buffers."""
    if len(pointer_parameters) != len(set(pointer_parameters)):
        raise RuntimeError("duplicate pointer in kernel restrict allowlist")
    buffers: dict[str, Any] = {}
    for parameter in pointer_parameters:
        if parameter in DEVICE_POINTER_BINDINGS:
            name = DEVICE_POINTER_BINDINGS[parameter]
            source = device
        else:
            name = parameter[:-4] if parameter.endswith("_all") else parameter
            source = workspace
        if name not in source:
            raise RuntimeError(
                f"kernel alias audit has no live binding for {parameter} -> {name}"
            )
        buffers[parameter] = source[name]
    spans = assert_nonoverlapping_allocations(buffers)
    return {
        "pointer_parameters": len(buffers),
        "nonempty_allocations": len(spans),
        "empty_allocations": len(buffers) - len(spans),
    }
