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
    ttfa_s: float = 0.0
    num_chunks: int = 1
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


def _collect_streaming_response(
    result_q: queue.Queue,
    deadline: float,
    chunk_timeout: float = 60,
):
    """Collect all streamed audio chunks for a single request.

    Returns (chunks, error_str).  Raises queue.Empty if *deadline*
    (absolute perf_counter timestamp) is exceeded while waiting.
    """
    chunks = []
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return chunks, "request timed out (total deadline exceeded)"
        wait = min(chunk_timeout, remaining)

        try:
            result, error = result_q.get(timeout=wait)
        except queue.Empty:
            return chunks, "request timed out (no response chunk within deadline)"

        if error:
            return chunks, str(error)

        audio = result.as_numpy("audio").squeeze()
        if audio.size > 0:
            chunks.append(audio)

        response = result.get_response()
        final_param = response.parameters.get("triton_final_response")
        if final_param and getattr(final_param, "bool_param", False):
            return chunks, None


def worker(
    worker_id: int,
    triton_url: str,
    texts: list[str],
    task_queue: list[int],
    queue_lock: threading.Lock,
    stats: BenchmarkStats,
    request_timeout: float = 120,
):
    result_q: queue.Queue = queue.Queue()
    first_chunk_time: list[float | None] = [None]

    def _on_response(result, error):
        if first_chunk_time[0] is None and result is not None:
            try:
                audio = result.as_numpy("audio").squeeze()
                if audio.size > 0:
                    first_chunk_time[0] = time.perf_counter()
            except Exception:
                pass
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

            first_chunk_time[0] = None
            t0 = time.perf_counter()
            deadline = t0 + request_timeout
            client.async_stream_infer(
                model_name=MODEL_NAME,
                inputs=inputs,
                outputs=outputs,
            )

            chunks, error_str = _collect_streaming_response(result_q, deadline)
            elapsed = time.perf_counter() - t0
            ttfa = (first_chunk_time[0] - t0) if first_chunk_time[0] else elapsed

            if error_str:
                if "timed out" in error_str:
                    # Drain any late-arriving chunks so they don't
                    # bleed into the next request on this stream.
                    client.stop_stream()
                    client.start_stream(callback=_on_response)

                stats.add(RequestResult(
                    text=text, num_samples=0, duration_s=elapsed,
                    ttfa_s=ttfa, error=error_str,
                ))
                print(f"[worker {worker_id:02d}] request {task_idx} FAILED "
                      f"({elapsed:.1f}s) — {error_str}")
            else:
                audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
                num_samples = len(audio)
                stats.add(RequestResult(
                    text=text, num_samples=num_samples, duration_s=elapsed,
                    ttfa_s=ttfa, num_chunks=len(chunks),
                ))
                print(
                    f"[worker {worker_id:02d}] request {task_idx} done — "
                    f"{num_samples / SAMPLE_RATE:.2f}s audio in {elapsed:.2f}s "
                    f"(TTFA: {ttfa:.3f}s, {len(chunks)} chunks)"
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
    parser.add_argument("--request-timeout", type=float, default=60,
                        help="Per-request timeout in seconds (default: 120)")
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
                args=(i, args.triton_url, texts, warmup_queue, warmup_lock,
                      warmup_stats, args.request_timeout),
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
            args=(i, args.triton_url, texts, task_queue, queue_lock,
                  stats, args.request_timeout),
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
        ttfas_ms = sorted(r.ttfa_s * 1000 for r in successes)
        mean_ttfa = sum(ttfas_ms) / len(ttfas_ms)
        avg_chunks = sum(r.num_chunks for r in successes) / len(successes)
        print()
        print("  Time to first audio (TTFA):")
        print(f"    mean:   {mean_ttfa:.1f} ms")
        print(f"    p95:    {ttfas_ms[int(len(ttfas_ms) * 0.95)]:.1f} ms")
        print()
        print(f"  Avg chunks per request: {avg_chunks:.1f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
