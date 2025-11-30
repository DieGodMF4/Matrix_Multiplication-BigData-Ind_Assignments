import argparse, time, csv, os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random, psutil

SEED = 403086

def generate_dense(n):
    rnd = random.Random(SEED)
    return [[rnd.random() for _ in range(n)] for __ in range(n)]

def worker_chunk(args):
    A_chunk, B, start_i = args
    C_chunk = []
    n = len(B)
    for idx, Ai in enumerate(A_chunk):
        Ci = [0.0]*n
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
        C_chunk.append(Ci)
    return (start_i, C_chunk)

def parallel_mul(A,B,p):
    n = len(A)
    chunk = n // p
    tasks = []
    for t in range(p):
        i0 = t*chunk
        i1 = n if t==p-1 else (t+1)*chunk
        tasks.append((A[i0:i1], B, i0))

    C = [[0.0]*n for _ in range(n)]

    with ProcessPoolExecutor(max_workers=p) as ex:
        for start_i, C_chunk in ex.map(worker_chunk, tasks):
            for off,row in enumerate(C_chunk):
                C[start_i+off] = row
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="+", type=int)
    ap.add_argument("-p","--threads", type=int, default=1)
    ap.add_argument("-r","--repeats", type=int, default=3)
    args = ap.parse_args()

    ts = f"results_python_parallel_{args.threads}threads.csv"
    with open(ts,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang","size","threads","repeats","avg_time_ms","speedup","efficiency","rss_mb"])

        T1 = {}
        proc = psutil.Process(os.getpid())

        for n in args.sizes:
            for r in range(args.repeats):
                A = generate_dense(n)
                B = generate_dense(n)

                t0 = time.perf_counter()
                _ = parallel_mul(A,B,args.threads)
                t1 = time.perf_counter()
                dt = (t1-t0)*1000

                # Store baseline
                if args.threads == 1:
                    T1[n] = dt

                if args.threads == 1:
                    speed = 1.0
                else:
                    base = T1.get(n, None)
                    speed = base/dt if base is not None else 1.0

                eff = speed / args.threads
                rss = proc.memory_info().rss/1e6

                w.writerow([
                    "python", n, args.threads, args.repeats,
                    f"{dt:.3f}", f"{speed:.3f}", f"{eff:.3f}", f"{rss:.2f}"
                ])

    print("Saved", ts)

if __name__ == "__main__":
    main()
