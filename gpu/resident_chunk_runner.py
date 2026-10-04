"""Resident exact GPU batch runner with fixed-size streamed shot buffers.

The selected decoder is constructed once against a padded fixed-size shot
model.  Its CUDA module, immutable graph arrays, and workspace remain resident.
Each chunk updates only shot offsets, detector ids, packed syndrome bits, and
the two syndrome hashes.  A short final chunk is padded with zero syndromes;
padding results are discarded before global shot-order assembly.

This layer deliberately does not decide whether a device result is exact.
It returns status dictionaries (or ``None`` after a backend failure) in source
order.  ``ExactGpuBackend`` owns strict status validation and CPU replay.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SHOT_BUFFER_NAMES = (
    "shot_offsets",
    "shot_detections",
    "shot_bits",
    "shot_hash1",
    "shot_hash2",
)


@dataclass(frozen=True)
class ChunkInput:
    source_start: int
    source_stop: int
    valid_shots: int
    padding_shots: int
    detection_count: int
    shot_offsets: np.ndarray
    shot_detections: np.ndarray
    shot_bits: np.ndarray
    shot_hash1: np.ndarray
    shot_hash2: np.ndarray


def _validate_source_shots(model: Any) -> tuple[int, int]:
    shots = int(model.metadata["shots"])
    detectors = int(model.metadata["num_detectors"])
    if shots < 0 or detectors < 0:
        raise ValueError("shot and detector counts must be nonnegative")
    offsets = np.asarray(model.shot_offsets)
    detections = np.asarray(model.shot_detections)
    words = (detectors + 63) // 64
    bits = np.asarray(model.shot_bits)
    hash1 = np.asarray(model.shot_hash1)
    hash2 = np.asarray(model.shot_hash2)
    if offsets.ndim != 1 or len(offsets) != shots + 1:
        raise ValueError("shot_offsets do not match metadata shots")
    if offsets.dtype.kind not in "iu" or offsets.dtype.itemsize > 8:
        raise ValueError("shot_offsets must use an integer dtype")
    if offsets.dtype.kind == "i" and bool(np.any(offsets < 0)):
        raise ValueError("shot_offsets must be nonnegative")
    wide = offsets.astype(np.uint64, copy=False)
    if int(wide[0]) != 0 or bool(np.any(wide[1:] < wide[:-1])):
        raise ValueError("shot_offsets must begin at zero and be monotonic")
    if detections.ndim != 1 or detections.dtype.kind not in "iu":
        raise ValueError("shot_detections must be a one-dimensional integer array")
    if int(wide[-1]) != len(detections):
        raise ValueError("shot_offsets terminal does not match shot_detections")
    if detections.dtype.kind == "i" and bool(np.any(detections < 0)):
        raise ValueError("shot_detections must be nonnegative")
    if detections.size:
        maximum_detection = int(detections.max(initial=0))
        if maximum_detection > np.iinfo(np.uint32).max:
            raise ValueError("shot_detections exceed the uint32 device ABI")
        if maximum_detection >= detectors:
            raise ValueError("shot_detections contain an out-of-range detector")
    if bits.shape != (shots, words):
        raise ValueError("shot_bits shape does not match shots and detectors")
    if hash1.shape != (shots,) or hash2.shape != (shots,):
        raise ValueError("shot hash arrays do not match metadata shots")
    for name, values in (
        ("shot_bits", bits), ("shot_hash1", hash1), ("shot_hash2", hash2),
    ):
        if values.dtype.kind not in "iu" or values.dtype.itemsize > 8:
            raise ValueError(f"{name} must use an integer dtype up to 64 bits")
        if values.dtype.kind == "i" and bool(np.any(values < 0)):
            raise ValueError(f"{name} must be nonnegative")
    return shots, words


def max_chunk_detections(model: Any, chunk_size: int) -> int:
    """Return the exact largest sequential chunk payload for this batch size."""
    shots, _ = _validate_source_shots(model)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    offsets = np.asarray(model.shot_offsets, dtype=np.uint64)
    maximum = 0
    for start in range(0, shots, chunk_size):
        stop = min(shots, start + chunk_size)
        maximum = max(maximum, int(offsets[stop] - offsets[start]))
    return maximum


def build_resident_chunk_model(
    model: Any, *, chunk_size: int, detection_capacity: int
) -> Any:
    """Make a shallow graph view with fixed, valid placeholder shot storage."""
    _shots, words = _validate_source_shots(model)
    chunk_size = int(chunk_size)
    detection_capacity = int(detection_capacity)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if detection_capacity < 0:
        raise ValueError("detection_capacity must be nonnegative")
    view = copy.copy(model)
    view.metadata = dict(model.metadata)
    view.metadata["shots"] = chunk_size
    view.metadata["source_shots"] = int(model.metadata["shots"])
    view.metadata["resident_chunk_size"] = chunk_size
    view.shot_offsets = np.zeros(chunk_size + 1, dtype=np.uint64)
    # Checked decoder constructors validate the complete CSR envelope before
    # the first chunk upload.  Assign the placeholder payload to the final row
    # so its terminal offset matches the fixed-capacity detection buffer.
    view.shot_offsets[-1] = detection_capacity
    view.shot_detections = np.zeros(detection_capacity, dtype=np.uint32)
    view.shot_bits = np.zeros((chunk_size, words), dtype=np.uint64)
    view.shot_hash1 = np.zeros(chunk_size, dtype=np.uint64)
    view.shot_hash2 = np.zeros(chunk_size, dtype=np.uint64)
    if hasattr(view, "reference"):
        empty = {
            "low_confidence": False,
            "predicted_errors_populated": True,
            "total_cost_bits": 0,
            "predicted_errors": [],
            "predictions": [],
        }
        view.reference = [dict(empty) for _ in range(chunk_size)]
    return view


def pack_chunk(
    model: Any,
    *,
    source_start: int,
    chunk_size: int,
    detection_capacity: int,
) -> ChunkInput:
    """Pack one sequential source range and exact zero-syndrome padding."""
    shots, words = _validate_source_shots(model)
    source_start = int(source_start)
    chunk_size = int(chunk_size)
    detection_capacity = int(detection_capacity)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if detection_capacity < 0:
        raise ValueError("detection_capacity must be nonnegative")
    if not 0 <= source_start <= shots:
        raise ValueError("source_start is outside the source batch")
    source_stop = min(shots, source_start + chunk_size)
    valid = source_stop - source_start
    source_offsets = np.asarray(model.shot_offsets, dtype=np.uint64)
    detection_begin = int(source_offsets[source_start])
    detection_end = int(source_offsets[source_stop])
    detection_count = detection_end - detection_begin
    if detection_count > detection_capacity:
        raise ValueError(
            f"chunk requires {detection_count} detections but resident capacity "
            f"is {detection_capacity}"
        )

    offsets = np.empty(chunk_size + 1, dtype=np.uint64)
    offsets[: valid + 1] = (
        source_offsets[source_start : source_stop + 1] - detection_begin
    )
    offsets[valid + 1 :] = detection_count
    detections = np.zeros(detection_capacity, dtype=np.uint32)
    if detection_count:
        detections[:detection_count] = np.asarray(
            model.shot_detections[detection_begin:detection_end],
            dtype=np.uint32,
        )
    bits = np.zeros((chunk_size, words), dtype=np.uint64)
    hash1 = np.zeros(chunk_size, dtype=np.uint64)
    hash2 = np.zeros(chunk_size, dtype=np.uint64)
    if valid:
        bits[:valid] = np.asarray(
            model.shot_bits[source_start:source_stop], dtype=np.uint64
        )
        hash1[:valid] = np.asarray(
            model.shot_hash1[source_start:source_stop], dtype=np.uint64
        )
        hash2[:valid] = np.asarray(
            model.shot_hash2[source_start:source_stop], dtype=np.uint64
        )
    return ChunkInput(
        source_start=source_start,
        source_stop=source_stop,
        valid_shots=valid,
        padding_shots=chunk_size - valid,
        detection_count=detection_count,
        shot_offsets=offsets,
        shot_detections=detections,
        shot_bits=bits,
        shot_hash1=hash1,
        shot_hash2=hash2,
    )


def _buffer_shape(buffer: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in buffer.shape)


def copy_to_device_buffer(buffer: Any, values: np.ndarray) -> None:
    """Copy into a real CuPy array or an injected array-compatible fake."""
    dtype = np.dtype(buffer.dtype)
    source = np.asarray(values)
    if dtype.kind in "iu" and source.dtype.kind not in "iu":
        raise TypeError(
            f"integer resident buffer dtype {dtype} requires integer input"
        )
    if source.dtype.kind in "iu" and dtype.kind in "iu" and source.size:
        bounds = np.iinfo(dtype)
        minimum = int(source.min(initial=0))
        maximum = int(source.max(initial=0))
        if minimum < bounds.min or maximum > bounds.max:
            raise OverflowError(
                f"resident buffer dtype {dtype} cannot losslessly represent "
                f"input range [{minimum}, {maximum}]"
            )
    converted = np.ascontiguousarray(values, dtype=dtype)
    if _buffer_shape(buffer) != converted.shape:
        raise ValueError(
            f"resident device buffer shape {_buffer_shape(buffer)} does not match "
            f"chunk shape {converted.shape}"
        )
    setter = getattr(buffer, "set", None)
    if callable(setter):
        setter(converted)
    else:
        buffer[...] = converted


class ResidentChunkedDecoder:
    """Construct once, stream fixed buffers, and assemble raw GPU attempts."""

    def __init__(
        self,
        model: Any,
        decoder_factory: Callable[..., Any],
        decoder_kwargs: Mapping[str, Any],
        *,
        chunk_size: int,
        detection_capacity: int | None = None,
        device_buffers: Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.source_shots, self.words = _validate_source_shots(model)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        computed_capacity = max_chunk_detections(model, self.chunk_size)
        self.detection_capacity = (
            computed_capacity
            if detection_capacity is None else int(detection_capacity)
        )
        if self.detection_capacity < computed_capacity:
            raise ValueError(
                f"detection_capacity {self.detection_capacity} is below required "
                f"{computed_capacity}"
            )
        self.chunk_model = build_resident_chunk_model(
            model,
            chunk_size=self.chunk_size,
            detection_capacity=self.detection_capacity,
        )
        self.decoder = decoder_factory(self.chunk_model, **dict(decoder_kwargs))
        buffers = device_buffers
        if buffers is None:
            buffers = getattr(self.decoder, "device", None)
        if not isinstance(buffers, Mapping):
            raise TypeError("resident decoder must expose a device buffer mapping")
        missing = [name for name in SHOT_BUFFER_NAMES if name not in buffers]
        if missing:
            raise ValueError(f"resident decoder is missing shot buffers: {missing}")
        self.device_buffers = {name: buffers[name] for name in SHOT_BUFFER_NAMES}
        expected = {
            "shot_offsets": self.chunk_model.shot_offsets,
            "shot_detections": self.chunk_model.shot_detections,
            "shot_bits": self.chunk_model.shot_bits,
            "shot_hash1": self.chunk_model.shot_hash1,
            "shot_hash2": self.chunk_model.shot_hash2,
        }
        for name, values in expected.items():
            if _buffer_shape(self.device_buffers[name]) != values.shape:
                raise ValueError(
                    f"resident {name} buffer has shape "
                    f"{_buffer_shape(self.device_buffers[name])}, expected {values.shape}"
                )
        self._disabled_error: str | None = None
        self.decode_calls = 0

    @property
    def resident(self) -> bool:
        return self.decoder is not None

    def _upload(self, chunk: ChunkInput) -> float:
        start = time.perf_counter()
        values = {
            "shot_offsets": chunk.shot_offsets,
            "shot_detections": chunk.shot_detections,
            "shot_bits": chunk.shot_bits,
            "shot_hash1": chunk.shot_hash1,
            "shot_hash2": chunk.shot_hash2,
        }
        for name in SHOT_BUFFER_NAMES:
            np.copyto(getattr(self.chunk_model, name), values[name])
            copy_to_device_buffer(self.device_buffers[name], values[name])
        return (time.perf_counter() - start) * 1000.0

    def decode(self) -> tuple[list[object], dict]:
        self.decode_calls += 1
        if self.source_shots == 0:
            return [], {
                "kernel_ms": 0.0,
                "upload_ms": 0.0,
                "status_counts": {},
                "chunk_size": self.chunk_size,
                "detection_capacity": self.detection_capacity,
                "chunk_count": 0,
                "chunks": [],
                "backend_error": self._disabled_error,
                "backend_failed_shots": [],
                "decode_call": self.decode_calls,
            }
        if self._disabled_error is not None:
            failed = list(range(self.source_shots))
            return [None] * self.source_shots, {
                "kernel_ms": 0.0,
                "upload_ms": 0.0,
                "status_counts": {},
                "chunk_size": self.chunk_size,
                "detection_capacity": self.detection_capacity,
                "chunk_count": 0,
                "chunks": [],
                "backend_error": self._disabled_error,
                "backend_failed_shots": failed,
                "decode_call": self.decode_calls,
            }

        decisions: list[object] = []
        chunk_rows: list[dict] = []
        failed_shots: list[int] = []
        total_kernel_ms = 0.0
        total_upload_ms = 0.0
        status_counts: Counter[int] = Counter()
        for chunk_index, source_start in enumerate(
            range(0, self.source_shots, self.chunk_size)
        ):
            chunk: ChunkInput | None = None
            upload_ms = 0.0
            try:
                chunk = pack_chunk(
                    self.model,
                    source_start=source_start,
                    chunk_size=self.chunk_size,
                    detection_capacity=self.detection_capacity,
                )
                upload_ms = self._upload(chunk)
                total_upload_ms += upload_ms
                result = self.decoder.decode()
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError("chunk decoder must return (decisions, stats)")
                chunk_decisions, chunk_stats = result
                if not isinstance(chunk_decisions, Sequence) or isinstance(
                    chunk_decisions, (str, bytes)
                ):
                    raise TypeError("chunk decisions must be a sequence")
                if len(chunk_decisions) != self.chunk_size:
                    raise ValueError(
                        f"chunk decoder returned {len(chunk_decisions)} decisions "
                        f"for resident batch {self.chunk_size}"
                    )
                if not isinstance(chunk_stats, dict):
                    raise TypeError("chunk stats must be a dictionary")
                kernel_ms = float(chunk_stats.get("kernel_ms", 0.0))
                valid_decisions = list(chunk_decisions[: chunk.valid_shots])
                decisions.extend(valid_decisions)
                for decision in valid_decisions:
                    if isinstance(decision, dict):
                        try:
                            status_counts[int(decision.get("status", 0))] += 1
                        except (TypeError, ValueError, OverflowError):
                            pass
                total_kernel_ms += kernel_ms
                chunk_rows.append({
                    "chunk": chunk_index,
                    "source_start": chunk.source_start,
                    "source_stop": chunk.source_stop,
                    "valid_shots": chunk.valid_shots,
                    "padding_shots": chunk.padding_shots,
                    "detection_count": chunk.detection_count,
                    "upload_ms": upload_ms,
                    "kernel_ms": kernel_ms,
                    "decoder_stats": chunk_stats,
                    "error": None,
                })
            except Exception as exc:
                self._disabled_error = f"{type(exc).__name__}: {exc}"
                # Everything before this chunk was already assembled.  The
                # current chunk and every later source shot must be replayed.
                failure_start = source_start
                if len(decisions) != failure_start:
                    # This is an internal assembly invariant, and falling back
                    # the whole source suffix remains the only exact route.
                    del decisions[failure_start:]
                remaining = self.source_shots - failure_start
                decisions.extend([None] * remaining)
                failed_shots.extend(range(failure_start, self.source_shots))
                chunk_rows.append({
                    "chunk": chunk_index,
                    "source_start": source_start,
                    "source_stop": (
                        chunk.source_stop if chunk is not None
                        else min(self.source_shots, source_start + self.chunk_size)
                    ),
                    "valid_shots": (
                        chunk.valid_shots if chunk is not None
                        else min(self.chunk_size, self.source_shots - source_start)
                    ),
                    "padding_shots": (
                        chunk.padding_shots if chunk is not None else None
                    ),
                    "detection_count": (
                        chunk.detection_count if chunk is not None else None
                    ),
                    "upload_ms": upload_ms,
                    "kernel_ms": None,
                    "decoder_stats": {},
                    "error": self._disabled_error,
                })
                break

        if len(decisions) != self.source_shots:
            raise RuntimeError(
                f"resident chunk assembly produced {len(decisions)} decisions "
                f"for {self.source_shots} source shots"
            )
        return decisions, {
            "kernel_ms": total_kernel_ms,
            "upload_ms": total_upload_ms,
            "status_counts": dict(sorted(status_counts.items())),
            "chunk_size": self.chunk_size,
            "detection_capacity": self.detection_capacity,
            "chunk_count": len(chunk_rows),
            "chunks": chunk_rows,
            "backend_error": self._disabled_error,
            "backend_failed_shots": failed_shots,
            "decode_call": self.decode_calls,
        }


__all__ = [
    "ChunkInput",
    "ResidentChunkedDecoder",
    "SHOT_BUFFER_NAMES",
    "build_resident_chunk_model",
    "copy_to_device_buffer",
    "max_chunk_detections",
    "pack_chunk",
]
