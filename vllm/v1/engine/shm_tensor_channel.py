# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-memory tensor channel for decode-step communication.

Provides bidirectional tensor transfer between client and core processes
using POSIX shared memory with flag-based signaling.  Designed for the
decode loop where the client writes custom inputs (e.g. shape [1, dim])
and the core writes back outputs each step.

Shared-memory layout::

    HEADER (264 bytes, aligned to 64):
      [0]       input_ready    (uint8, client -> core)
      [1]       output_ready   (uint8, core -> client)
      [4:8]     request_id_len (uint32 LE)
      [8:264]   request_id     (UTF-8, max 256 bytes)

    INPUT DATA REGION:  contiguous buffer for input tensors
    OUTPUT DATA REGION: contiguous buffer for output tensors

Tensor data offsets are computed deterministically from the specs
passed at construction time, so both sides agree on the layout.
"""

from __future__ import annotations

import math
import multiprocessing.shared_memory as shm
import struct
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INPUT_READY_OFF = 0
_OUTPUT_READY_OFF = 1
_REQ_ID_LEN_OFF = 4
_REQ_ID_OFF = 8
_REQ_ID_MAX = 256
_HEADER_SIZE = _REQ_ID_OFF + _REQ_ID_MAX  # 264
_ALIGN = 64


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
           #logger.info(
           #     ">>>>>[SHM] CREATED channel name=%s request_id=%s "
           #     "total_size=%d input_region_off=%d output_region_off=%d "
           #     "input_slots=%s output_slots=%s buf_id=%d",
           #     name, request_id, total_size,
           #     self._input_region_off, self._output_region_off,
           #     {k: (off, s.shape, s.dtype)
           #      for k, (off, s) in self._input_slots.items()},
           #     {k: (off, s.shape, s.dtype)
           #      for k, (off, s) in self._output_slots.items()},
           #     id(self._buf),
           # )
        else:
            self._shm = shm.SharedMemory(name=name, create=False)
            self._buf = self._shm.buf
            #logger.info(
            #    ">>>>>[SHM] OPENED channel name=%s request_id=%s "
            #    "shm_size=%d input_region_off=%d output_region_off=%d "
            #    "input_slots=%s output_slots=%s buf_id=%d",
            #    name, request_id, self._shm.size,
            #    self._input_region_off, self._output_region_off,
            #    {k: (off, s.shape, s.dtype)
            #     for k, (off, s) in self._input_slots.items()},
            #    {k: (off, s.shape, s.dtype)
            #     for k, (off, s) in self._output_slots.items()},
            #    id(self._buf),
            #)

    # ── internal helpers ───────────────────────────────────────────

    def _write_tensor(
        self,
        region_off: int,
        local_off: int,
        spec: TensorSpec,
        tensor: torch.Tensor,
    ) -> None:
        assert tensor.shape == spec.shape, (
            f"Shape mismatch for '{spec.name}': "
            f"expected {spec.shape}, got {tensor.shape}"
        )
        assert tensor.dtype == spec.dtype, (
            f"Dtype mismatch for '{spec.name}': "
            f"expected {spec.dtype}, got {tensor.dtype}"
        )
        t = tensor.detach().cpu().clone().contiguous()
        raw = bytes(t.untyped_storage())
        off = region_off + local_off
        self._buf[off : off + spec.nbytes] = raw[: spec.nbytes]

    def _read_tensor(
        self, region_off: int, local_off: int, spec: TensorSpec
    ) -> torch.Tensor:
        off = region_off + local_off
        raw = bytearray(self._buf[off : off + spec.nbytes])
        return (
            torch.frombuffer(raw, dtype=spec.dtype)
            .reshape(spec.shape)
            .clone()
        )

    # ── Client -> Core ─────────────────────────────────────────────

    def write_input(self, name: str, tensor: torch.Tensor) -> None:
        local_off, spec = self._input_slots[name]
        self._write_tensor(self._input_region_off, local_off, spec, tensor)

    def write_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        for name, tensor in inputs.items():
            self.write_input(name, tensor)

    def signal_input_ready(self) -> None:
        #logger.info(">>>>>[SHM %s] signal_input_ready: setting flag at offset %d "
        #             "(shm_name=%s, buf_id=%d)",
        #             self.request_id, _INPUT_READY_OFF, self.name, id(self._buf))
        self._buf[_INPUT_READY_OFF] = 1
        #logger.info(">>>>>[SHM %s] signal_input_ready: flag value after set = %d",
        #             self.request_id, int(self._buf[_INPUT_READY_OFF]))

    def check_input_ready(self) -> bool:
        val = bool(self._buf[_INPUT_READY_OFF])
        #if val:
        #    logger.info(">>>>>[SHM %s] check_input_ready: flag=1 (shm_name=%s, "
        #                 "buf_id=%d)", self.request_id, self.name, id(self._buf))
        return val

    def consume_input(self) -> dict[str, torch.Tensor]:
        """Read all input tensors and clear the input_ready flag."""
        #logger.info(">>>>>[SHM %s] consume_input: clearing input_ready flag",
        #             self.request_id)
        self._buf[_INPUT_READY_OFF] = 0
        result = {
            name: self._read_tensor(self._input_region_off, off, spec)
            for name, (off, spec) in self._input_slots.items()
        }
        #logger.info(">>>>>[SHM %s] consume_input: read %d tensors: %s",
        #             self.request_id, len(result),
        #             {k: (v.shape, v.dtype) for k, v in result.items()})
        return result

    # ── Core -> Client ─────────────────────────────────────────────

    def can_write_outputs(self, outputs: dict[str, torch.Tensor]) -> bool:
        """Return True if every tensor matches its output spec shape/dtype."""
        for name, tensor in outputs.items():
            entry = self._output_slots.get(name)
            if entry is None:
                #logger.info(">>>>>[SHM %s] can_write_outputs: no slot for '%s'",
                #             self.request_id, name)
                return False
            _, spec = entry
            if tensor.shape != spec.shape or tensor.dtype != spec.dtype:
                #logger.info(">>>>>[SHM %s] can_write_outputs: mismatch for '%s': "
                #             "expected shape=%s dtype=%s, got shape=%s dtype=%s",
                #             spec.shape, spec.dtype,
                #             tensor.shape, tensor.dtype)
                return False
        return True

    def write_output(self, name: str, tensor: torch.Tensor) -> None:
        local_off, spec = self._output_slots[name]
        self._write_tensor(self._output_region_off, local_off, spec, tensor)

    def write_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        #logger.info(">>>>>[SHM %s] write_outputs: writing %d tensors: %s",
        #             self.request_id, len(outputs),
        #             {k: (v.shape, v.dtype) for k, v in outputs.items()})
        for name, tensor in outputs.items():
            self.write_output(name, tensor)

    def signal_output_ready(self) -> None:
        #logger.info(">>>>>[SHM %s] signal_output_ready: setting flag at offset %d "
        #             "(shm_name=%s, buf_id=%d)",
        #             self.request_id, _OUTPUT_READY_OFF, self.name, id(self._buf))
        self._buf[_OUTPUT_READY_OFF] = 1
        #logger.info(">>>>>[SHM %s] signal_output_ready: flag value after set = %d",
        #             self.request_id, int(self._buf[_OUTPUT_READY_OFF]))

    def check_output_ready(self) -> bool:
        val = bool(self._buf[_OUTPUT_READY_OFF])
        return val

    def consume_output(self) -> dict[str, torch.Tensor]:
        """Read all output tensors and clear the output_ready flag."""
        #logger.info(">>>>>[SHM %s] consume_output: clearing output_ready flag",
        #             self.request_id)
        self._buf[_OUTPUT_READY_OFF] = 0
        result = {
            name: self._read_tensor(self._output_region_off, off, spec)
            for name, (off, spec) in self._output_slots.items()
        }
        #logger.info(">>>>>[SHM %s] consume_output: read %d tensors: %s",
        #             self.request_id, len(result),
        #             {k: (v.shape, v.dtype) for k, v in result.items()})
        return result

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
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
