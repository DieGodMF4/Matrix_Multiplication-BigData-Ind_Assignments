import argparse, random, time, statistics, csv, datetime, os
import psutil, tracemalloc

BASE_SEED = 403086

def gen_dense(n, seed):
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(n)] for __ in range(n)]

def gen_sparse_csr(n, density, seed):
    """Return CSR (row_ptr, col_idx, vals) for an n x n matrix with given density (0..1)."""
    rnd = random.Random(seed)
    row_ptr = [0]
    col_idx, vals = [], []
    nnz_target = int(n*n*density)
    # simple row-wise fill: nnz per row ≈ density * n
    per_row = max(0, nnz_target // n)
    leftover = nnz_target - per_row*n
    for i in range(n):
        k = per_row + (1 if i < leftover else 0)
        # choose k distinct columns
        cols = rnd.sample(range(n), k) if k <= n else list(range(n))
        cols.sort()
        for j in cols:
            col_idx.append(j)
            vals.append(rnd.random())
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx, vals

def matmul_basic(A, B):
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C

def matmul_transposed(A, B):
    n = len(A)
    # pretranspose B to improve locality
    BT = [[B[j][i] for j in range(n)] for i in range(n)]
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for j in range(n):
            btj = BT[j]
            s = 0.0
            for k in range(n):
                s += Ai[k] * btj[k]
            Ci[j] = s
    return C

def matmul_blocked(A, B, BS):
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    # 3-level tiled multiply
    for ii in range(0, n, BS):
        for kk in range(0, n, BS):
            for jj in range(0, n, BS):
                i_max = min(ii+BS, n)
                k_max = min(kk+BS, n)
                j_max = min(jj+BS, n)
                for i in range(ii, i_max):
                    Ci = C[i]; Ai = A[i]
                    for k in range(kk, k_max):
                        aik = Ai[k]; Bk = B[k]
                        for j in range(jj, j_max):
                            Ci[j] += aik * Bk[j]
    return C

def spmm_csr_dense(row_ptr, col_idx, vals, B):
    """C = A_sparse(CSR) @ B_dense; C is dense."""
    n = len(B[0])        # columns of B
    m = len(row_ptr)-1   # rows of A
    C = [[0.0]*n for _ in range(m)]
    for i in range(m):
        start, end = row_ptr[i], row_ptr[i+1]
        Ci = C[i]
        for p in range(start, end):
            k = col_idx[p]
            aik = vals[p]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C

def mem_stats(proc):
    rss = proc.memory_info().rss / 1e6
    vms = proc.memory_info().vms / 1e6
    return rss, vms

def run_once(n, algo, BS, density):
    proc = psutil.Process(os.getpid())
    if algo == "sparse":
        row_ptr, col_idx, vals = gen_sparse_csr(n, density, BASE_SEED)
        B = gen_dense(n, BASE_SEED+1)
        t0 = time.perf_counter(); _ = spmm_csr_dense(row_ptr, col_idx, vals, B); t1 = time.perf_counter()
    elif algo == "blocked":
        A = gen_dense(n, BASE_SEED); B = gen_dense(n, BASE_SEED+1)
        t0 = time.perf_counter(); _ = matmul_blocked(A, B, BS); t1 = time.perf_counter()
    elif algo == "transposed":
        A = gen_dense(n, BASE_SEED); B = gen_dense(n, BASE_SEED+1)
        t0 = time.perf_counter(); _ = matmul_transposed(A, B); t1 = time.perf_counter()
    else:
        A = gen_dense(n, BASE_SEED); B = gen_dense(n, BASE_SEED+1)
        t0 = time.perf_counter(); _ = matmul_basic(A, B); t1 = time.perf_counter()
    return (t1 - t0) * 1000.0

def main():
    ap = argparse.ArgumentParser(description="Optimized Matrix Multiply (Python)")
    ap.add_argument("sizes", nargs="+", type=int)
    ap.add_argument("--algo", choices=["basic","blocked","transposed","sparse"], default="blocked")
    ap.add_argument("-r","--repeats", type=int, default=3)
    ap.add_argument("--block", type=int, default=64, help="Block size for blocked algo")
    ap.add_argument("--density", type=float, default=0.05, help="Non-zero density for sparse (0..1)")
    args = ap.parse_args()

    tracemalloc.start()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = f"results_python_opt_{args.algo}_{ts}.csv"
    with open(outname,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang","algo","size","repeats","avg_time_ms","psutil_rss_mb","psutil_vms_mb","tracemalloc_peak_mib","extra"])
        for n in args.sizes:
            times = [run_once(n, args.algo, args.block, args.density) for _ in range(args.repeats)]
            avg_ms = statistics.mean(times)
            current, peak = tracemalloc.get_traced_memory()
            proc = psutil.Process(os.getpid()); rss, vms = mem_stats(proc)
            extra = {"block": args.block, "density": args.density}
            w.writerow(["python", args.algo, n, args.repeats, f"{avg_ms:.3f}", f"{rss:.2f}", f"{vms:.2f}", f"{peak/(1024*1024):.2f}", extra])
    print(f"! Saved {outname}")

if __name__ == "__main__":
    random.seed(BASE_SEED)
    main()
