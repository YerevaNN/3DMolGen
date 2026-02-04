#!/usr/bin/env python3
"""Quick script to regenerate COV vs samples plots without std bands."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- CONFIG ----
# CSV_PATH = Path("outputs/gen_results/20260126_233340_qw600_pre_binned_filtered_4e_top_p_r4_08_t08_distinct/cov_vs_samples_data.csv")
CSV_PATH = Path("outputs/gen_results/20260124_124554_qw600_pre_binned_filtered_4e_top_k_r3_70_t13_distinct/cov_vs_samples_data_s256.csv")

THRESHOLD = 0.75
OUTPUT_DIR = CSV_PATH.parent
DPI = 300
# ----------------

df = pd.read_csv(CSV_PATH)
df_thresh = df[np.isclose(df["threshold"], THRESHOLD)].sort_values("n_samples")

fig, ax = plt.subplots(figsize=(8, 6))
x = df_thresh["n_samples"].values

# COV-R (blue) - line only, no fill
ax.plot(x, df_thresh["cov_r_mean"].values, "o-", color="#1f77b4",
        linewidth=2, markersize=6, label="COV-R")

# COV-P (red) - line only, no fill
ax.plot(x, df_thresh["cov_p_mean"].values, "s-", color="#d62728",
        linewidth=2, markersize=6, label="COV-P")

ax.set_xscale("log", base=2)
ax.set_xlabel("Number of Samples", fontsize=12)
ax.set_ylabel("Coverage", fontsize=12)
ax.set_title(f"Coverage vs Number of Samples (threshold = {THRESHOLD} Å)", fontsize=12)
ax.set_ylim(0, 1)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in x])

plt.tight_layout()

# Save
threshold_str = f"{THRESHOLD:.2f}".replace(".", "_")
for fmt in ["png", "pdf"]:
    out_path = OUTPUT_DIR / f"cov_vs_samples_t{threshold_str}_no_std.{fmt}"
    fig.savefig(out_path, dpi=DPI if fmt == "png" else None, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
