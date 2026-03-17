#!/usr/bin/env python3
"""
Benchmark script for Qwen3-TTS Triton server (decoupled mode, gRPC).

Spawns N concurrent workers that send TTS requests in parallel.
Texts are randomly sampled from a provided text file (one line per utterance).

Usage:
    python benchmark_server.py --text-file texts.txt --num-requests 100 --num-workers 8
    python benchmark_server.py --text-file texts.txt --num-requests 50 --num-workers 4 --triton-url localhost:8001
"""

import argparse
import queue
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import tritonclient.grpc as grpcclient

SAMPLE_RATE = 24_000
MODEL_NAME = "qwen3_tts"


@dataclass
class RequestResult:
    text: str
    num_samples: int
    duration_s: float
    error: str | None = None


@dataclass
class BenchmarkStats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: list[RequestResult] = field(default_factory=list)

    def add(self, result: RequestResult):
        with self.lock:
            self.results.append(result)


def _make_inputs(text: str, language: str = "english"):
    text_input = grpcclient.InferInput("text", [1, 1], "BYTES")
    text_input.set_data_from_numpy(np.array([[text]], dtype=object))

    lang_input = grpcclient.InferInput("language", [1, 1], "BYTES")
    lang_input.set_data_from_numpy(np.array([[language]], dtype=object))

    outputs = [grpcclient.InferRequestedOutput("audio")]
    return [text_input, lang_input], outputs


def worker(
    worker_id: int,
    triton_url: str,
    texts: list[str],
    task_queue: list[int],
    queue_lock: threading.Lock,
    stats: BenchmarkStats,
):
    result_q: queue.Queue = queue.Queue()

    def _on_response(result, error):
        result_q.put((result, error))

    client = grpcclient.InferenceServerClient(url=triton_url)
    client.start_stream(callback=_on_response)

    try:
        while True:
            with queue_lock:
                if not task_queue:
                    return
                task_idx = task_queue.pop()

            text = random.choice(texts)
            inputs, outputs = _make_inputs(text)

            t0 = time.perf_counter()
            client.async_stream_infer(
                model_name=MODEL_NAME,
                inputs=inputs,
                outputs=outputs,
            )

            result, error = result_q.get(timeout=120)
            elapsed = time.perf_counter() - t0

            if error:
                stats.add(RequestResult(text=text, num_samples=0, duration_s=elapsed, error=str(error)))
                print(f"[worker {worker_id:02d}] request {task_idx} FAILED — {error}")
            else:
                audio = result.as_numpy("audio").squeeze()
                num_samples = len(audio)
                stats.add(RequestResult(text=text, num_samples=num_samples, duration_s=elapsed))
                print(
                    f"[worker {worker_id:02d}] request {task_idx} done — "
                    f"{num_samples / SAMPLE_RATE:.2f}s audio in {elapsed:.2f}s"
                )
    finally:
        client.stop_stream()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS Triton server")
    parser.add_argument("--text-file", required=True, help="Path to file with one text per line")
    parser.add_argument("--num-requests", type=int, required=True, help="Total number of requests to send")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC endpoint (default: localhost:8001)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase (3 requests per worker)")
    args = parser.parse_args()

    with open(args.text_file) as f:
        texts = [line.split('\t', 1)[1].strip() for line in f if line.strip()]

    if not texts:
        print(f"ERROR: no non-empty lines found in {args.text_file}")
        return

    print(f"Loaded {len(texts)} texts from {args.text_file}")
    print(f"Sending {args.num_requests} requests with {args.num_workers} workers to {args.triton_url}")
    print("-" * 70)

    WARMUP_PER_WORKER = 3
    if not args.no_warmup:
        total_warmup = args.num_workers * WARMUP_PER_WORKER
        print(f"Warmup: {total_warmup} requests ({WARMUP_PER_WORKER} per worker) ...")
        warmup_queue = list(range(total_warmup))
        warmup_lock = threading.Lock()
        warmup_stats = BenchmarkStats()

        warmup_threads = []
        for i in range(args.num_workers):
            t = threading.Thread(
                target=worker,
                args=(i, args.triton_url, texts, warmup_queue, warmup_lock, warmup_stats),
            )
            t.start()
            warmup_threads.append(t)
        for t in warmup_threads:
            t.join()
        print("Warmup complete.")
        print("-" * 70)

    task_queue = list(range(args.num_requests))
    queue_lock = threading.Lock()
    stats = BenchmarkStats()

    wall_start = time.perf_counter()

    threads = []
    for i in range(args.num_workers):
        t = threading.Thread(
            target=worker,
            args=(i, args.triton_url, texts, task_queue, queue_lock, stats),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    wall_elapsed = time.perf_counter() - wall_start

    successes = [r for r in stats.results if r.error is None]
    failures = [r for r in stats.results if r.error is not None]

    total_audio_samples = sum(r.num_samples for r in successes)
    total_audio_seconds = total_audio_samples / SAMPLE_RATE

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Total requests sent:      {args.num_requests}")
    print(f"  Successful:               {len(successes)}")
    print(f"  Failed:                   {len(failures)}")
    print(f"  Concurrent workers:       {args.num_workers}")
    print()
    print(f"  Wall-clock time:          {wall_elapsed:.2f} s")
    print(f"  Total audio synthesized:  {total_audio_seconds:.2f} s")
    print(f"  Real-time factor (RTF):   {total_audio_seconds / wall_elapsed:.2f}x")
    print(f"  Throughput:               {len(successes) / wall_elapsed:.2f} requests/s")

    if successes:
        latencies = [r.duration_s for r in successes]
        latencies.sort()
        print()
        print("  Per-request latency:")
        print(f"    min:    {latencies[0]:.3f} s")
        print(f"    median: {latencies[len(latencies) // 2]:.3f} s")
        print(f"    p90:    {latencies[int(len(latencies) * 0.9)]:.3f} s")
        print(f"    p99:    {latencies[int(len(latencies) * 0.99)]:.3f} s")
        print(f"    max:    {latencies[-1]:.3f} s")

    print("=" * 70)


if __name__ == "__main__":
    main()
