import argparse, random, time, statistics
import psutil, tracemalloc, os
import csv, datetime, os

"""
Basic O(n^3) matrix multiply benchmark in Python.
Usage: python mm_baseline.py <sizes...> [-r REPEATS]
Example: python mm_baseline.py 64 128 256 -r 5
"""

BASE_SEED = 403086

def gen_matrix(n, seed):
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(n)] for __ in range(n)]

def matmul_basic(A, B):
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C

def bench(n, repeats):
    proc = psutil.Process(os.getpid())
    times_ms = []
    rss_peak_mb = 0.0  # we will sample after each repetition
    tracemalloc.start()

    for r in range(repeats):
        A = gen_matrix(n, seed=BASE_SEED + r)
        B = gen_matrix(n, seed=BASE_SEED + 1 + r)

        t0 = time.perf_counter()
        _ = matmul_basic(A, B)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

        rss_now_mb = proc.memory_info().rss / 1e6
        if rss_now_mb > rss_peak_mb:
            rss_peak_mb = rss_now_mb

    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_ms = statistics.mean(times_ms)
    rss_now_mb = proc.memory_info().rss / 1e6
    vms_now_mb = proc.memory_info().vms / 1e6

    # Report both psutil (RSS/VMS) and tracemalloc peak (Python heap)
    mem_info = {
        "psutil_rss_mb_now": f"{rss_now_mb:.2f}",
        "psutil_vms_mb_now": f"{vms_now_mb:.2f}",
        "psutil_peak_rss_mb": f"{rss_peak_mb:.2f}",
        "tracemalloc_peak_mib": f"{tm_peak / (1024*1024):.2f}",
    }
    return avg_ms, mem_info

def main():
    ap = argparse.ArgumentParser(description="Basic O(n^3) matrix multiply benchmark (Python).")
    ap.add_argument("sizes", nargs="+", type=int, help="Square sizes, e.g. 64 128 256")
    ap.add_argument("-r", "--repeats", type=int, default=3, help="Repetitions per size (default: 3)")
    args = ap.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = f"results_python_{timestamp}.csv"
    header = ["lang","size","repeats","avg_time_ms","psutil_rss_mb_now","psutil_vms_mb_now","psutil_peak_rss_mb","tracemalloc_peak_mib"]

    with open(outname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for n in args.sizes:
            avg_ms, mem = bench(n, args.repeats)
            writer.writerow([
                "python", n, args.repeats, f"{avg_ms:.3f}",
                mem["psutil_rss_mb_now"], mem["psutil_vms_mb_now"],
                mem["psutil_peak_rss_mb"], mem["tracemalloc_peak_mib"]
            ])
    print(f"Results saved to {outname}")


if __name__ == "__main__":
    random.seed(BASE_SEED)
    main()
