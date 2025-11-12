"""
Clustered Matrix Multiplication Visualizer
------------------------------------------
Produces 2x2 plots: Dense algorithms (basic, blocked, transposed) vs Sparse.
Color = language, Marker = algorithm.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
DATA_DIRS = [
    Path("../C Implementation"),
    Path("../Java Implementation"),
    Path("../Python Implementation")
]
OUT_DIR = Path(".")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.4, "font.size": 11})

marker_map = {
    "basic": "o",
    "blocked": "s",
    "transposed": "^",
    "sparse": "X",
}
color_map = {
    "c": "tab:blue",
    "java": "tab:orange",
    "python": "tab:green",
}

# === LOAD ===
dfs = []
for d in DATA_DIRS:
    for f in d.glob("results_*opt_*.csv"):
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
            print(f"Loaded: {f.name}")
        except Exception as e:
            print(f"~Skipping {f.name}: {e}")

if not dfs:
    raise SystemExit("!! No CSV files found.")

data = pd.concat(dfs, ignore_index=True)
for c in data.columns:
    if data[c].dtype == object:
        data[c] = data[c].astype(str).str.replace(",", ".")
for c in data.columns[2:]:
    data[c] = pd.to_numeric(data[c], errors="ignore")
if "algo" not in data.columns:
    data["algo"] = "unknown"

# --- Separate dense and sparse ---
dense_algos = ["basic", "blocked", "transposed"]
dense = data[data["algo"].isin(dense_algos)]
sparse = data[data["algo"].isin(["sparse"])]

# --- Memory column mapping ---
mem_cols_by_lang = {
    "c": "rss_peak_mb",
    "java": "heap_now_mb",
    "python": "psutil_rss_mb",
}

def plot_group(df, title_suffix, fname_prefix):
    """Generate time and memory plots for given subset (dense or sparse)."""
    # TIME
    plt.figure(figsize=(8, 6))
    for (lang, algo), grp in df.groupby(["lang", "algo"]):
        color = color_map.get(lang.lower(), "gray")
        marker = marker_map.get(algo.lower(), "o")
        plt.plot(
            grp["size"], grp["avg_time_ms"],
            marker=marker, linewidth=1.8, markersize=7,
            color=color,
            label=f"{lang.upper()} - {algo}"
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix size (N)")
    plt.ylabel("Average execution time (ms)")
    plt.title(f"Optimized Matrix Multiplication ({title_suffix}) - Performance")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname_prefix}_time.png", dpi=300)
    print(f"Saved: {fname_prefix}_time.png")

    # MEMORY
    plt.figure(figsize=(8, 6))
    for (lang, algo), grp in df.groupby(["lang", "algo"]):
        color = color_map.get(lang.lower(), "gray")
        marker = marker_map.get(algo.lower(), "o")
        col = mem_cols_by_lang.get(lang.lower())
        if col not in grp.columns:
            candidates = [c for c in grp.columns if "peak" in c.lower()]
            if candidates: col = candidates[0]
            else:
                print(f"~No memory column for {lang} ({algo})")
                continue
        plt.plot(
            grp["size"], grp[col],
            marker=marker, linewidth=1.8, markersize=7,
            color=color,
            label=f"{lang.upper()} - {algo}"
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix size (N)")
    plt.ylabel("Peak memory (MB)")
    plt.title(f"Optimized Matrix Multiplication ({title_suffix}) - Memory")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname_prefix}_memory.png", dpi=300)
    print(f"Saved: {fname_prefix}_memory.png")

# === PLOT ===
if not dense.empty:
    plot_group(dense, "Dense (basic / blocked / transposed)", "mm_perf_dense")

if not sparse.empty:
    plot_group(sparse, "Sparse (CSR × Dense)", "mm_perf_sparse")

print("\nTwo clusters plotted successfully.")
