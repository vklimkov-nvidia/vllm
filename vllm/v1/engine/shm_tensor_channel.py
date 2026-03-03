# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-memory tensor channel for decode-step communication.

Provides bidirectional tensor transfer between client and core processes
using POSIX shared memory with flag-based signaling.  Designed for the
decode loop where the client writes custom inputs (e.g. shape [1, dim])
and the core writes back outputs each step.

Uses Linux futex(2) for efficient cross-process waiting: the waiting
side sleeps in the kernel until the signaling side writes the flag and
calls FUTEX_WAKE, yielding ~1-5 µs wake latency with zero CPU usage.

Shared-memory layout::

    HEADER (272 bytes, aligned to 64):
      [0:4]     input_ready    (uint32 LE, client -> core, futex word)
      [4:8]     output_ready   (uint32 LE, core -> client, futex word)
      [8:12]    request_id_len (uint32 LE)
      [12:268]  request_id     (UTF-8, max 256 bytes)

    INPUT DATA REGION:  contiguous buffer for input tensors
    OUTPUT DATA REGION: contiguous buffer for output tensors

Tensor data offsets are computed deterministically from the specs
passed at construction time, so both sides agree on the layout.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import math
import multiprocessing.shared_memory as shm
import struct
import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INPUT_READY_OFF = 0   # uint32, 4-byte aligned for futex
_OUTPUT_READY_OFF = 4  # uint32, 4-byte aligned for futex
_REQ_ID_LEN_OFF = 8
_REQ_ID_OFF = 12
_REQ_ID_MAX = 256
_HEADER_SIZE = _REQ_ID_OFF + _REQ_ID_MAX  # 268
_ALIGN = 64

# ── Linux futex helpers ───────────────────────────────────────────
_SYS_FUTEX = 202  # x86_64
_FUTEX_WAIT = 0
_FUTEX_WAKE = 1

# ctypes used only for syscall and address lookup; struct used for flag I/O
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _futex_wait(addr: int, expected: int,
                timeout_s: float | None) -> int:
    """Sleep until the uint32 at *addr* differs from *expected*.
    Returns 0 on wake, -1 on error (check errno: EAGAIN=ready, ETIMEDOUT=timeout).
    """
    if timeout_s is not None:
        sec = int(timeout_s)
        nsec = int((timeout_s - sec) * 1_000_000_000)
        ts = (ctypes.c_long * 2)(sec, nsec)
        ret = _libc.syscall(
            ctypes.c_long(_SYS_FUTEX), ctypes.c_void_p(addr),
            ctypes.c_int(_FUTEX_WAIT), ctypes.c_int(expected),
            ctypes.byref(ts),
        )
    else:
        ret = _libc.syscall(
            ctypes.c_long(_SYS_FUTEX), ctypes.c_void_p(addr),
            ctypes.c_int(_FUTEX_WAIT), ctypes.c_int(expected), None,
        )
    return ret


def _futex_wake(addr: int, count: int = 1) -> int:
    """Wake up to *count* threads sleeping on the futex at *addr*."""
    return _libc.syscall(
        ctypes.c_long(_SYS_FUTEX), ctypes.c_void_p(addr),
        ctypes.c_int(_FUTEX_WAKE), ctypes.c_int(count),
    )


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


@dataclass(frozen=True)
class TensorSpec:
    """Describes a named tensor slot with fixed shape and dtype."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty(
            0, dtype=self.dtype
        ).element_size()


def decode_step_tensor_specs(
    custom_specs: list,
    model_dtype: torch.dtype,
) -> list["TensorSpec"]:
    """Convert ``CustomInputSpec`` list to ``TensorSpec`` list for a
    single decode step (batch size 1)."""
    from vllm.config.model import CustomInputSpec

    result: list[TensorSpec] = []
    for spec in custom_specs:
        assert isinstance(spec, CustomInputSpec)
        dtype = spec.get_torch_dtype() or model_dtype
        shape = (1,) if spec.dim is None else (1, spec.dim)
        result.append(TensorSpec(name=spec.name, shape=shape, dtype=dtype))
    return result


class SharedMemoryTensorChannel:
    """Bidirectional tensor channel via POSIX shared memory.

    The creator (client) passes ``create=True`` and the opener (core)
    passes ``create=False``.  Both sides must supply identical specs.
    """

    def __init__(
        self,
        name: str,
        request_id: str,
        input_specs: list[TensorSpec],
        output_specs: list[TensorSpec],
        create: bool = True,
    ):
        self.name = name
        self.request_id = request_id
        self._is_creator = create
        self._wait_input_stats: list[float] = []

        # Build offset tables: name -> (local_byte_offset, TensorSpec)
        self._input_slots: dict[str, tuple[int, TensorSpec]] = {}
        self._output_slots: dict[str, tuple[int, TensorSpec]] = {}

        input_data_size = 0
        for spec in input_specs:
            self._input_slots[spec.name] = (input_data_size, spec)
            input_data_size += spec.nbytes

        output_data_size = 0
        for spec in output_specs:
            self._output_slots[spec.name] = (output_data_size, spec)
            output_data_size += spec.nbytes

        self._input_region_off = _align_up(_HEADER_SIZE, _ALIGN)
        self._output_region_off = _align_up(
            self._input_region_off + input_data_size, _ALIGN
        )
        total_size = max(self._output_region_off + output_data_size, 1)

        if create:
            try:
                old = shm.SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self._shm = shm.SharedMemory(
                name=name, create=True, size=total_size
            )
            self._buf = self._shm.buf
            self._buf[:_HEADER_SIZE] = b"\x00" * _HEADER_SIZE
            rid = request_id.encode("utf-8")[:_REQ_ID_MAX]
            struct.pack_into("<I", self._buf, _REQ_ID_LEN_OFF, len(rid))
            self._buf[_REQ_ID_OFF : _REQ_ID_OFF + len(rid)] = rid
        else:
            self._shm = shm.SharedMemory(name=name, create=False)
            self._buf = self._shm.buf

        # Cache raw addresses for futex syscalls.  One-time ctypes use to get
        # the address; flag I/O uses struct for lower overhead in the hot path.
        _input_view = (ctypes.c_char * 4).from_buffer(self._shm.buf,
                                                      _INPUT_READY_OFF)
        _output_view = (ctypes.c_char * 4).from_buffer(self._shm.buf,
                                                        _OUTPUT_READY_OFF)
        self._input_futex_addr = ctypes.addressof(_input_view)
        self._output_futex_addr = ctypes.addressof(_output_view)

        # Pre-create zero-copy tensor views backed by the shm buffer.
        # Reads/writes then only need a single copy_() or clone().
        self._input_shm_views: dict[str, torch.Tensor] = {}
        for name, (local_off, spec) in self._input_slots.items():
            off = self._input_region_off + local_off
            self._input_shm_views[name] = torch.frombuffer(
                self._buf, dtype=spec.dtype,
                count=math.prod(spec.shape), offset=off,
            ).reshape(spec.shape)

        self._output_shm_views: dict[str, torch.Tensor] = {}
        for name, (local_off, spec) in self._output_slots.items():
            off = self._output_region_off + local_off
            self._output_shm_views[name] = torch.frombuffer(
                self._buf, dtype=spec.dtype,
                count=math.prod(spec.shape), offset=off,
            ).reshape(spec.shape)

    # ── Client -> Core ─────────────────────────────────────────────

    def write_input(self, name: str, tensor: torch.Tensor) -> None:
        view = self._input_shm_views[name]
        t = tensor.detach().cpu()
        assert t.shape == view.shape, (
            f"Shape mismatch for '{name}': "
            f"expected {view.shape}, got {t.shape}"
        )
        assert t.dtype == view.dtype, (
            f"Dtype mismatch for '{name}': "
            f"expected {view.dtype}, got {t.dtype}"
        )
        view.copy_(t)

    def write_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        for name, tensor in inputs.items():
            self.write_input(name, tensor)

    def signal_input_ready(self) -> None:
        """Client: set input_ready flag and wake the core via futex."""
        struct.pack_into("<I", self._buf, _INPUT_READY_OFF, 1)
        _futex_wake(self._input_futex_addr)

    def check_input_ready(self) -> bool:
        return struct.unpack_from("<I", self._buf, _INPUT_READY_OFF)[0] != 0

    def wait_input_ready(self, timeout_s: float | None = None) -> bool:
        """Core: block until input_ready is set, or timeout.

        Returns True if input is ready, False on timeout.
        Uses a single futex_wait; kernel returns EAGAIN if value already set.
        """
        t0 = time.perf_counter()
        ret = _futex_wait(self._input_futex_addr, 0, timeout_s)
        self._wait_input_stats.append(time.perf_counter() - t0)
        if ret == 0:
            return True  # woken by client
        if ret == -1 and ctypes.get_errno() == errno.EAGAIN:
            return True  # value already set (race), no sleep needed
        return False  # timeout or error

    def consume_input(self) -> dict[str, torch.Tensor]:
        """Read all input tensors and clear the input_ready flag.

        Returns views (no clone) since the consumer only reads via copy_();
        the client will not overwrite until after the next output round-trip.
        """
        struct.pack_into("<I", self._buf, _INPUT_READY_OFF, 0)
        return dict(self._input_shm_views)

    # ── Core -> Client ─────────────────────────────────────────────

    def can_write_outputs(self, outputs: dict[str, torch.Tensor]) -> bool:
        """Return True if every tensor matches its output spec shape/dtype."""
        for name, tensor in outputs.items():
            entry = self._output_slots.get(name)
            if entry is None:
                return False
            _, spec = entry
            if tensor.shape != spec.shape or tensor.dtype != spec.dtype:
                return False
        return True

    def write_output(self, name: str, tensor: torch.Tensor) -> None:
        view = self._output_shm_views[name]
        t = tensor.detach().cpu()
        assert t.shape == view.shape, (
            f"Shape mismatch for '{name}': "
            f"expected {view.shape}, got {t.shape}"
        )
        assert t.dtype == view.dtype, (
            f"Dtype mismatch for '{name}': "
            f"expected {view.dtype}, got {t.dtype}"
        )
        view.copy_(t)

    def write_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        for name, tensor in outputs.items():
            self.write_output(name, tensor)

    def signal_output_ready(self) -> None:
        """Core: set output_ready flag and wake the client via futex."""
        struct.pack_into("<I", self._buf, _OUTPUT_READY_OFF, 1)
        _futex_wake(self._output_futex_addr)

    def check_output_ready(self) -> bool:
        return struct.unpack_from("<I", self._buf, _OUTPUT_READY_OFF)[0] != 0

    def wait_output_ready(self, timeout_s: float | None = None) -> bool:
        """Client: block until output_ready is set, or timeout.

        Returns True if output is ready, False on timeout.
        Uses a single futex_wait; kernel returns EAGAIN if value already set.
        """
        ret = _futex_wait(self._output_futex_addr, 0, timeout_s)
        if ret == 0:
            return True  # woken by core
        if ret == -1 and ctypes.get_errno() == errno.EAGAIN:
            return True  # value already set (race), no sleep needed
        return False  # timeout or error

    def consume_output(self) -> dict[str, torch.Tensor]:
        """Read all output tensors and clear the output_ready flag.

        Returns views (no clone) since the consumer only reads; the core
        will not overwrite until after the next forward pass.
        """
        struct.pack_into("<I", self._buf, _OUTPUT_READY_OFF, 0)
        return dict(self._output_shm_views)

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        if hasattr(self, "_wait_input_stats") and self._wait_input_stats:
            count = len(self._wait_input_stats)
            total_ms = sum(self._wait_input_stats) * 1000
            avg_ms = total_ms / count
            min_ms = min(self._wait_input_stats) * 1000
            max_ms = max(self._wait_input_stats) * 1000
            print(f">>>[SharedMemoryTensorChannel] wait_input_ready stats: "
                  f"count={count}, avg={avg_ms:.3f}ms, "
                  f"min={min_ms:.3f}ms, max={max_ms:.3f}ms", flush=True)

        # Release tensor views before closing the mmap.
        self._input_shm_views.clear()
        self._output_shm_views.clear()
        if hasattr(self, "_shm"):
            self._shm.close()
            if self._is_creator:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
