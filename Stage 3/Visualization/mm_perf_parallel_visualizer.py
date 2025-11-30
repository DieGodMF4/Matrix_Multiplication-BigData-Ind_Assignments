#!/usr/bin/env python3
"""
Parallel Matrix Multiplication Visualizer - Assignment 3
Loads all parallel CSVs (C, Java, Python) and plots:
 - Speedup vs Threads
 - Efficiency vs Threads
 - Execution Time vs Threads

One plot per matrix size (256, 512, 1024, ...)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========================== CONFIG ===================================

DATA_DIRS = [
    Path("../C Implementation"),
    Path("../Java Implementation"),
    Path("../Python Implementation")
]

OUT_DIR = Path(".")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 12
})

COLOR_MAP = {
    "c": "tab:blue",
    "java": "tab:orange",
    "python": "tab:green"
}

MARKER_MAP = {
    "c": "o",
    "java": "s",
    "python": "^"
}

# ========================== LOAD DATA =================================

dfs = []
for d in DATA_DIRS:
    for f in d.glob("results_*parallel_*threads.csv"):
        try:
            df = pd.read_csv(f)
            df["source"] = f.name
            dfs.append(df)
            print(f"Loaded: {f.name}")
        except Exception as e:
            print(f"! Could not load {f}: {e}")

if not dfs:
    raise SystemExit("!! No parallel CSV files found.")

data = pd.concat(dfs, ignore_index=True)

# Normalize data types
for col in ["avg_time_ms", "speedup", "efficiency"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Clean language column just in case
data["lang"] = data["lang"].str.lower()

print("\nLoaded parallel data:\n", data.head(), "\n")

# ========================== UNIQUE SIZES ===============================
sizes = sorted(data["size"].unique())

# ========================== PLOT FUNCTIONS ============================

def plot_speedup_for_size(n):
    plt.figure(figsize=(8,6))
    subset = data[data["size"] == n]

    for lang, grp in subset.groupby("lang"):
        col = COLOR_MAP.get(lang, "black")
        marker = MARKER_MAP.get(lang, "x")
        plt.plot(
            grp["threads"], grp["speedup"],
            marker=marker, markersize=8, linewidth=2,
            color=col, label=lang.upper()
        )
    
    plt.xlabel("Threads (p)")
    plt.ylabel("Speedup (T1 / Tp)")
    plt.title(f"Speedup vs Threads (n = {n})")
    plt.xticks(sorted(subset["threads"].unique()))
    plt.legend()
    plt.tight_layout()
    fname = OUT_DIR / f"parallel_speedup_{n}.png"
    plt.savefig(fname, dpi=300)
    print(f"Saved: {fname}")


def plot_efficiency_for_size(n):
    plt.figure(figsize=(8,6))
    subset = data[data["size"] == n]

    for lang, grp in subset.groupby("lang"):
        col = COLOR_MAP.get(lang, "black")
        marker = MARKER_MAP.get(lang, "x")
        plt.plot(
            grp["threads"], grp["efficiency"],
            marker=marker, markersize=8, linewidth=2,
            color=col, label=lang.upper()
        )
    
    plt.xlabel("Threads (p)")
    plt.ylabel("Efficiency (Speedup / p)")
    plt.title(f"Efficiency vs Threads (n = {n})")
    plt.xticks(sorted(subset["threads"].unique()))
    plt.legend()
    plt.tight_layout()
    fname = OUT_DIR / f"parallel_efficiency_{n}.png"
    plt.savefig(fname, dpi=300)
    print(f"Saved: {fname}")


def plot_time_for_size(n):
    plt.figure(figsize=(8,6))
    subset = data[data["size"] == n]

    for lang, grp in subset.groupby("lang"):
        col = COLOR_MAP.get(lang, "black")
        marker = MARKER_MAP.get(lang, "x")
        plt.plot(
            grp["threads"], grp["avg_time_ms"],
            marker=marker, markersize=8, linewidth=2,
            color=col, label=lang.upper()
        )
    
    plt.xscale("linear")
    plt.yscale("log")
    plt.xlabel("Threads (p)")
    plt.ylabel("Execution Time (ms, log scale)")
    plt.title(f"Execution Time vs Threads (n = {n})")
    plt.xticks(sorted(subset["threads"].unique()))
    plt.legend()
    plt.tight_layout()
    fname = OUT_DIR / f"parallel_time_{n}.png"
    plt.savefig(fname, dpi=300)
    print(f"- Saved: {fname}")

# ========================== GENERATE PLOTS ============================

for n in sizes:
    plot_speedup_for_size(n)
    plot_efficiency_for_size(n)
    plot_time_for_size(n)

print("\nAll parallel plots completed!")
