"""
plot_realtime_tradeoffs.py

Simple helper to pretty-print the realtime_benchmark.txt
and (optionally) extend to multiple configurations.
"""

import os


RESULTS_DIR = os.path.join("research_extensions", "results")
BENCH_PATH = os.path.join(RESULTS_DIR, "realtime_benchmark.txt")


def main():
    if not os.path.exists(BENCH_PATH):
        raise FileNotFoundError(f"Benchmark file not found at: {BENCH_PATH}")

    print("\n⚙️ Real-time Benchmark Summary:\n")
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            print(line.strip())


if __name__ == "__main__":
    main()

