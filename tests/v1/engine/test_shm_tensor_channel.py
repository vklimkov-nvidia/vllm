# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedMemoryTensorChannel.

Run with: pytest tests/v1/engine/test_shm_tensor_channel.py -v
"""

from __future__ import annotations

import multiprocessing
import time

import pytest
import torch

from vllm.v1.engine.shm_tensor_channel import (
    SharedMemoryTensorChannel,
    TensorSpec,
)

INPUT_SPECS = [
    TensorSpec(name="hidden", shape=(1, 128), dtype=torch.float32),
    TensorSpec(name="mask", shape=(1, 128), dtype=torch.bool),
]
OUTPUT_SPECS = [
    TensorSpec(name="logits", shape=(1, 64), dtype=torch.float32),
]


class TestTensorSpec:
    def test_nbytes_float32(self):
        spec = TensorSpec(name="x", shape=(2, 3), dtype=torch.float32)
        assert spec.nbytes == 2 * 3 * 4

    def test_nbytes_bool(self):
        spec = TensorSpec(name="m", shape=(1, 10), dtype=torch.bool)
        assert spec.nbytes == 10

    def test_nbytes_float16(self):
        spec = TensorSpec(name="h", shape=(4, 8), dtype=torch.float16)
        assert spec.nbytes == 4 * 8 * 2

    def test_nbytes_int64(self):
        spec = TensorSpec(name="ids", shape=(1, 32), dtype=torch.int64)
        assert spec.nbytes == 1 * 32 * 8

    def test_nbytes_scalar(self):
        spec = TensorSpec(name="s", shape=(1,), dtype=torch.float32)
        assert spec.nbytes == 4


class TestSameProcess:
    """Tests with create=True and create=False in the same process."""

    def test_flags_initially_clear(self):
        ch = SharedMemoryTensorChannel(
            "test_flags", "req-0", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        try:
            assert not ch.check_input_ready()
            assert not ch.check_output_ready()
        finally:
            ch.close()

    def test_roundtrip_inputs(self):
        client = SharedMemoryTensorChannel(
            "test_rt_in", "req-1", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        core = SharedMemoryTensorChannel(
            "test_rt_in", "req-1", INPUT_SPECS, OUTPUT_SPECS, create=False
        )
        try:
            hidden = torch.randn(1, 128)
            mask = torch.ones(1, 128, dtype=torch.bool)

            client.write_inputs({"hidden": hidden, "mask": mask})
            client.signal_input_ready()

            assert core.check_input_ready()
            result = core.consume_input()
            assert not core.check_input_ready()

            torch.testing.assert_close(result["hidden"], hidden)
            torch.testing.assert_close(result["mask"], mask)
        finally:
            core.close()
            client.close()

    def test_roundtrip_outputs(self):
        client = SharedMemoryTensorChannel(
            "test_rt_out", "req-2", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        core = SharedMemoryTensorChannel(
            "test_rt_out", "req-2", INPUT_SPECS, OUTPUT_SPECS, create=False
        )
        try:
            logits = torch.randn(1, 64)
            core.write_outputs({"logits": logits})
            core.signal_output_ready()

            assert client.check_output_ready()
            result = client.consume_output()
            assert not client.check_output_ready()

            torch.testing.assert_close(result["logits"], logits)
        finally:
            core.close()
            client.close()

    def test_write_single_tensor(self):
        client = SharedMemoryTensorChannel(
            "test_single", "req-3", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        core = SharedMemoryTensorChannel(
            "test_single", "req-3", INPUT_SPECS, OUTPUT_SPECS, create=False
        )
        try:
            hidden = torch.randn(1, 128)
            mask = torch.zeros(1, 128, dtype=torch.bool)

            client.write_input("hidden", hidden)
            client.write_input("mask", mask)
            client.signal_input_ready()

            result = core.consume_input()
            torch.testing.assert_close(result["hidden"], hidden)
            torch.testing.assert_close(result["mask"], mask)
        finally:
            core.close()
            client.close()

    def test_multiple_steps(self):
        client = SharedMemoryTensorChannel(
            "test_multi", "req-4", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        core = SharedMemoryTensorChannel(
            "test_multi", "req-4", INPUT_SPECS, OUTPUT_SPECS, create=False
        )
        try:
            for i in range(5):
                hidden = torch.full((1, 128), float(i))
                mask = torch.ones(1, 128, dtype=torch.bool)
                client.write_inputs({"hidden": hidden, "mask": mask})
                client.signal_input_ready()

                inputs = core.consume_input()
                torch.testing.assert_close(inputs["hidden"], hidden)

                logits = torch.full((1, 64), float(i * 10))
                core.write_outputs({"logits": logits})
                core.signal_output_ready()

                outputs = client.consume_output()
                torch.testing.assert_close(outputs["logits"], logits)
        finally:
            core.close()
            client.close()

    def test_shape_mismatch_raises(self):
        client = SharedMemoryTensorChannel(
            "test_shape_err", "req-5", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        try:
            with pytest.raises(AssertionError, match="Shape mismatch"):
                client.write_input("hidden", torch.randn(2, 128))
        finally:
            client.close()

    def test_dtype_mismatch_raises(self):
        client = SharedMemoryTensorChannel(
            "test_dtype_err", "req-6", INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        try:
            with pytest.raises(AssertionError, match="Dtype mismatch"):
                client.write_input(
                    "hidden", torch.randn(1, 128).to(torch.float16)
                )
        finally:
            client.close()

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.int32, torch.int64, torch.uint8],
    )
    def test_various_dtypes(self, dtype):
        specs_in = [TensorSpec(name="x", shape=(2, 4), dtype=dtype)]
        specs_out = [TensorSpec(name="y", shape=(1, 8), dtype=dtype)]
        name = f"test_dtype_{dtype}".replace(".", "_")

        client = SharedMemoryTensorChannel(
            name, "req-dt", specs_in, specs_out, create=True
        )
        core = SharedMemoryTensorChannel(
            name, "req-dt", specs_in, specs_out, create=False
        )
        try:
            if dtype.is_floating_point:
                x = torch.randn(2, 4, dtype=dtype)
            else:
                x = torch.randint(0, 127, (2, 4), dtype=dtype)

            client.write_inputs({"x": x})
            client.signal_input_ready()
            result = core.consume_input()
            torch.testing.assert_close(result["x"], x)
        finally:
            core.close()
            client.close()


# ── Cross-process tests ──────────────────────────────────────────


def _core_worker(
    shm_name: str,
    request_id: str,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    num_steps: int,
    result_queue: multiprocessing.Queue,
):
    """Simulates the core side in a subprocess."""
    ch = SharedMemoryTensorChannel(
        shm_name, request_id, input_specs, output_specs, create=False
    )
    try:
        for step in range(num_steps):
            deadline = time.monotonic() + 5.0
            while not ch.check_input_ready():
                if time.monotonic() > deadline:
                    result_queue.put(("timeout", step))
                    return
                time.sleep(0.0001)

            inputs = ch.consume_input()
            total = sum(t.float().sum().item() for t in inputs.values())
            logits = torch.full(output_specs[0].shape, total)
            ch.write_outputs({"logits": logits})
            ch.signal_output_ready()

        result_queue.put(("ok", num_steps))
    finally:
        ch.close()


class TestCrossProcess:
    """Cross-process tests using multiprocessing.Process."""

    def test_roundtrip(self):
        shm_name = "test_xproc_rt"
        request_id = "req-xp"
        num_steps = 3

        client = SharedMemoryTensorChannel(
            shm_name, request_id, INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_core_worker,
            args=(
                shm_name, request_id, INPUT_SPECS, OUTPUT_SPECS,
                num_steps, result_queue,
            ),
        )
        proc.start()
        try:
            for step in range(num_steps):
                hidden = torch.full((1, 128), float(step + 1))
                mask = torch.ones(1, 128, dtype=torch.bool)
                client.write_inputs({"hidden": hidden, "mask": mask})
                client.signal_input_ready()

                deadline = time.monotonic() + 5.0
                while not client.check_output_ready():
                    assert time.monotonic() < deadline, "Timed out"
                    time.sleep(0.0001)

                outputs = client.consume_output()
                expected = hidden.sum().item() + mask.float().sum().item()
                torch.testing.assert_close(
                    outputs["logits"],
                    torch.full((1, 64), expected),
                )

            status, count = result_queue.get(timeout=5)
            assert status == "ok"
            assert count == num_steps
        finally:
            proc.join(timeout=5)
            client.close()

    def test_many_steps(self):
        """Stress test with more steps."""
        shm_name = "test_xproc_many"
        request_id = "req-xp-many"
        num_steps = 20

        client = SharedMemoryTensorChannel(
            shm_name, request_id, INPUT_SPECS, OUTPUT_SPECS, create=True
        )
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_core_worker,
            args=(
                shm_name, request_id, INPUT_SPECS, OUTPUT_SPECS,
                num_steps, result_queue,
            ),
        )
        proc.start()
        try:
            for step in range(num_steps):
                hidden = torch.randn(1, 128)
                mask = torch.ones(1, 128, dtype=torch.bool)
                client.write_inputs({"hidden": hidden, "mask": mask})
                client.signal_input_ready()

                deadline = time.monotonic() + 5.0
                while not client.check_output_ready():
                    assert time.monotonic() < deadline, "Timed out"
                    time.sleep(0.0001)

                outputs = client.consume_output()
                expected = hidden.sum().item() + mask.float().sum().item()
                torch.testing.assert_close(
                    outputs["logits"],
                    torch.full((1, 64), expected),
                    atol=1e-3,
                    rtol=1e-3,
                )

            status, count = result_queue.get(timeout=5)
            assert status == "ok"
            assert count == num_steps
        finally:
            proc.join(timeout=5)
            client.close()
