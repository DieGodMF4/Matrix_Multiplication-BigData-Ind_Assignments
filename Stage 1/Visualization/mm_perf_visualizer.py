"""
Matrix Multiplication Performance Visualizer
---------------------------------------------
Reads all results_*.csv files (C / Java / Python)
and plots:
  1. Execution time vs matrix size (log-log)
  2. Peak memory usage vs matrix size (log-log)
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
OUT_DIR = Path(".")  # save plots here

# === LOAD CSVs ===
dfs = []
for d in DATA_DIRS:
    for f in d.glob("results_*.csv"):
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
            print(f"Loaded: {f}")
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

if not dfs:
    raise SystemExit("No CSV files found in implementation folders.")

data = pd.concat(dfs, ignore_index=True)

# === CLEANUP ===
# Replace commas with dots (for any locale issues)
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].astype(str).str.replace(",", ".")
# Convert numeric columns
for col in data.columns[2:]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\nData loaded successfully.")

# === PLOT 1: Execution Time vs Matrix Size ===
plt.figure(figsize=(8, 6))
for lang, grp in data.groupby("lang"):
    plt.plot(
        grp["size"], grp["avg_time_ms"],
        marker="o", linewidth=2, label=lang.upper()
    )

plt.title("Matrix Multiplication Performance Comparison")
plt.xlabel("Matrix size (N)")
plt.ylabel("Average execution time (ms)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()

time_plot = OUT_DIR / "mm_perf_time.png"
plt.savefig(time_plot, dpi=300)
print(f"Saved: {time_plot}")

# === PLOT 2: Peak Memory vs Matrix Size ===
plt.figure(figsize=(8, 6))

# Define best memory column per language
mem_map = {
    "c": "rss_peak_mb",
    "java": "heap_peak_mb",
    "python": "psutil_peak_rss_mb"  # or "tracemalloc_peak_mib"
}

for lang, grp in data.groupby("lang"):
    col = mem_map.get(lang.lower())
    if col in grp.columns:
        plt.plot(
            grp["size"], grp[col],
            marker="s", linewidth=2, label=f"{lang.upper()} ({col})"
        )
    else:
        print(f"No memory column found for {lang}")

plt.title("Peak Memory Usage Comparison")
plt.xlabel("Matrix size (N)")
plt.ylabel("Peak memory (MB)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()

mem_plot = OUT_DIR / "mm_perf_memory.png"
plt.savefig(mem_plot, dpi=300)
print(f"Saved: {mem_plot}")


print("\nVisualization complete.")
