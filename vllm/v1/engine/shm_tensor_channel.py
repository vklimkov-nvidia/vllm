# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-memory tensor channel for decode-step communication.

Provides bidirectional tensor transfer between client and core processes
using POSIX shared memory with C++-accelerated signaling.  Designed for
the decode loop where the client writes custom inputs (e.g. shape
[N, dim] where N = num_output_tokens_per_step, default 1) and the
core writes back outputs each step.

All flag operations use C++ atomics with acquire/release ordering,
futex syscalls are issued directly from C++ (no ctypes overhead),
the GIL is released during all waits and memory copies, and tensor
writes are batched into a single GIL-free memcpy+signal sequence.

Requires the ``_shm_channel_cpp`` C++ extension (built from
``csrc/shm_channel.cpp``).

Shared-memory layout::

    HEADER (268 bytes, aligned to 64):
      [0:4]     input_ready    (uint32, client -> core, futex word)
      [4:8]     output_ready   (uint32, core -> client, futex word)
      [8:12]    request_id_len (uint32)
      [12:268]  request_id     (UTF-8, max 256 bytes)

    INPUT DATA REGION:  contiguous buffer for input tensors
    OUTPUT DATA REGION: contiguous buffer for output tensors

Tensor data offsets are computed deterministically from the specs
passed at construction time, so both sides agree on the layout.
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing.shared_memory as shm
import struct
import threading
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import _posixshmem  # type: ignore[attr-defined]

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# ── C++ extension (optional at import time) ──────────────────────
try:
    import vllm._shm_channel_cpp as _cpp
except ImportError:
    _cpp = None  # type: ignore[assignment]

_INPUT_READY_OFF = 0
_OUTPUT_READY_OFF = 4
_REQ_ID_LEN_OFF = 8
_REQ_ID_OFF = 12
_REQ_ID_MAX = 256
_HEADER_SIZE = _REQ_ID_OFF + _REQ_ID_MAX  # 268
_ALIGN = 64


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


_shm_create_lock = threading.Lock()


def _create_shm_untracked(
    name: str, create: bool, size: int = 0,
) -> shm.SharedMemory:
    """Create a SharedMemory without registering it with the resource tracker.

    We manage close()/unlink() ourselves, so tracker bookkeeping is
    unnecessary and causes spurious warnings at shutdown.
    """
    with _shm_create_lock:
        orig = resource_tracker.register
        resource_tracker.register = lambda *args, **kwargs: None
        try:
            return shm.SharedMemory(name=name, create=create, size=size)
        finally:
            resource_tracker.register = orig


def _prepare_copy_descs(
    shm_views: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
) -> list[tuple[int, int, int]]:
    """Build (dst_ptr, src_ptr, nbytes) descriptors for C++ batch_copy.

    Caller must provide contiguous tensors with matching shape/dtype.
    """
    return [
        (shm_views[name].data_ptr(), tensor.data_ptr(),
         shm_views[name].nbytes)
        for name, tensor in tensors.items()
    ]


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
    num_output_tokens_per_step: int = 1,
) -> list["TensorSpec"]:
    """Convert ``CustomIOSpec`` list to ``TensorSpec`` list for a
    single decode step.

    When *num_output_tokens_per_step* > 1 (e.g. FastConformer producing
    multiple frames per step), the leading dimension of every tensor is
    set to that value instead of 1.
    """
    from vllm.config.model import CustomIOSpec

    n = max(num_output_tokens_per_step, 1)
    result: list[TensorSpec] = []
    for spec in custom_specs:
        assert isinstance(spec, CustomIOSpec)
        dtype = spec.get_torch_dtype() or model_dtype
        shape = (n,) if spec.dim is None else (n, spec.dim)
        result.append(TensorSpec(name=spec.name, shape=shape, dtype=dtype))
    return result


class SharedMemoryTensorChannel:
    """Bidirectional tensor channel via POSIX shared memory + C++ IPC.

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
        if _cpp is None:
            raise ImportError(
                "vllm._shm_channel_cpp is required for "
                "SharedMemoryTensorChannel. "
                "Build it with: pip install -e .  (or rebuild vllm)"
            )

        self.name = name
        self.request_id = request_id
        self._is_creator = create
        self.decode_started = False

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
            shm_name = "/" + name if not name.startswith("/") else name
            try:
                _posixshmem.shm_unlink(shm_name)
            except FileNotFoundError:
                pass
            self._shm = _create_shm_untracked(
                name, create=True, size=total_size,
            )
            self._buf = self._shm.buf
            self._buf[:_HEADER_SIZE] = b"\x00" * _HEADER_SIZE
            rid = request_id.encode("utf-8")[:_REQ_ID_MAX]
            struct.pack_into("<I", self._buf, _REQ_ID_LEN_OFF, len(rid))
            self._buf[_REQ_ID_OFF : _REQ_ID_OFF + len(rid)] = rid
        else:
            self._shm = _create_shm_untracked(name, create=False)
            self._buf = self._shm.buf

        self._closed = False

        # Raw addresses for C++ atomic/futex ops.
        # ctypes views kept alive so the buffer export stays valid.
        self._input_futex_view = (ctypes.c_char * 4).from_buffer(
            self._shm.buf, _INPUT_READY_OFF)
        self._output_futex_view = (ctypes.c_char * 4).from_buffer(
            self._shm.buf, _OUTPUT_READY_OFF)
        self._input_futex_addr = ctypes.addressof(self._input_futex_view)
        self._output_futex_addr = ctypes.addressof(self._output_futex_view)

        # Prefault + mlock via C++ to prevent page eviction stalls.
        base = ctypes.addressof(
            (ctypes.c_char * 1).from_buffer(self._shm.buf, 0))
        if not _cpp.prefault_mlock(base, self._shm.size):
            logger.warning(
                "mlock(%s, %d) failed; shm pages may be evicted "
                "causing sporadic copy stalls. Run: ulimit -l unlimited",
                self._shm.name, self._shm.size,
            )

        # Pre-create zero-copy tensor views backed by the shm buffer.
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

        # Cached (name, dst_ptr, nbytes) per input slot — these never
        # change, letting decode_step avoid per-step dict lookups and
        # .data_ptr()/.nbytes calls on the shm views.
        self._input_copy_cache: list[tuple[str, int, int]] = [
            (name, view.data_ptr(), view.nbytes)
            for name, view in self._input_shm_views.items()
        ]

    # ── Client -> Core ─────────────────────────────────────────────

    def write_input(self, name: str, tensor: torch.Tensor) -> None:
        view = self._input_shm_views[name]
        t = tensor.detach()
        assert t.shape == view.shape and t.dtype == view.dtype
        t = t.contiguous()
        _cpp.batch_copy([(view.data_ptr(), t.data_ptr(), view.nbytes)])

    def write_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        _cpp.batch_copy(_prepare_copy_descs(
            self._input_shm_views, inputs))

    def signal_input_ready(self) -> None:
        """Client: atomic set input_ready + FUTEX_WAKE."""
        _cpp.set_flag_and_wake(self._input_futex_addr)

    def check_input_ready(self) -> bool:
        return _cpp.check_flag(self._input_futex_addr)

    def wait_input_ready(self, timeout_s: float | None = None) -> bool:
        """Core: spin + futex wait with GIL released.

        Returns True if input is ready, False on timeout.
        """
        return _cpp.wait_flag(
            self._input_futex_addr,
            -1.0 if timeout_s is None else timeout_s)

    def consume_input(self) -> dict[str, torch.Tensor]:
        """Clear input_ready and return zero-copy shm tensor views."""
        _cpp.clear_flag(self._input_futex_addr)
        return dict(self._input_shm_views)

    # ── Core -> Client ─────────────────────────────────────────────

    def can_write_outputs(self, outputs: dict[str, torch.Tensor]) -> bool:
        """Return True if every tensor matches its output spec."""
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
        t = tensor.detach()
        assert t.shape == view.shape and t.dtype == view.dtype
        t = t.contiguous()
        _cpp.batch_copy([(view.data_ptr(), t.data_ptr(), view.nbytes)])

    def write_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        _cpp.batch_copy(_prepare_copy_descs(
            self._output_shm_views, outputs))

    def signal_output_ready(self) -> None:
        """Core: atomic set output_ready + FUTEX_WAKE."""
        _cpp.set_flag_and_wake(self._output_futex_addr)

    def check_output_ready(self) -> bool:
        return _cpp.check_flag(self._output_futex_addr)

    def wait_output_ready(self, timeout_s: float | None = None) -> bool:
        """Client: spin + futex wait with GIL released.

        Returns True if output is ready, False on timeout.
        """
        return _cpp.wait_flag(
            self._output_futex_addr,
            -1.0 if timeout_s is None else timeout_s)

    def consume_output(self) -> dict[str, torch.Tensor]:
        """Clear output_ready and return zero-copy shm tensor views."""
        _cpp.clear_flag(self._output_futex_addr)
        return dict(self._output_shm_views)

    # ── Synchronous decode step ───────────────────────────────────

    def decode_step(
        self,
        custom_inputs: dict[str, torch.Tensor],
        timeout: float = 10.0,
    ) -> dict[str, torch.Tensor]:
        """Write inputs, signal core, block until output, return results.

        Runs entirely in the calling thread with no asyncio involvement.
        The caller **must** provide contiguous tensors with matching
        shape and dtype.
        """
        copies = [
            (dst_ptr, custom_inputs[name].data_ptr(), nbytes)
            for name, dst_ptr, nbytes in self._input_copy_cache
        ]
        _cpp.batch_copy_and_signal(copies, self._input_futex_addr)

        if not _cpp.wait_and_clear_flag(self._output_futex_addr, timeout):
            raise TimeoutError(
                f"Decode step timed out after {timeout}s")

        return dict(self._output_shm_views)

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True

        self._input_shm_views.clear()
        self._output_shm_views.clear()
        self._input_futex_view = None
        self._output_futex_view = None
        self._buf = None

        if hasattr(self, "_shm"):
            name = self._shm._name
            try:
                self._shm.close()
            except BufferError:
                logger.warning(
                    "SharedMemoryTensorChannel %s: could not close shm — "
                    "outstanding buffer references exist (torch profiler?)",
                    name,
                )
            if self._is_creator:
                try:
                    _posixshmem.shm_unlink(name)
                except FileNotFoundError:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
