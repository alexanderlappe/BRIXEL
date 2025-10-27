import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

# -------- Paths --------
ours_path = r"C:\Users\Alex\Documents\Uni_Data\DENSE\eval\results\complexity\complexity_256.csv"
baseline_path = r"C:\Users\Alex\Documents\Uni_Data\DENSE\eval\results\complexity\complexity_1024.csv"

# -------- Target baseline models (and order) --------
baseline_models = [
    "dinov3_vits16",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
]

# -------- Load --------
ours = pd.read_csv(ours_path)
base = pd.read_csv(baseline_path)

required_cols = {"model", "gflops", "peak_mem_mib", "wall_time_s"}
for name, df in [("yours (256)", ours), ("baseline (1024)", base)]:
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV is missing columns: {missing}")

# -------- Filter to relevant rows --------
base_f = base[base["model"].isin(baseline_models)].copy()

# Expect "<baseline>_adapter" in the 256 CSV
ours_expected_names = [m + "_adapter" for m in baseline_models]
ours_f = ours[ours["model"].isin(ours_expected_names)].copy()

def strip_adapter(name: str) -> str:
    return name[:-8] if isinstance(name, str) and name.endswith("_adapter") else name

base_f["model_base"] = base_f["model"]
ours_f["model_base"] = ours_f["model"].map(strip_adapter)

missing_in_ours = sorted(set(baseline_models) - set(ours_f["model_base"]))
missing_in_base = sorted(set(baseline_models) - set(base_f["model_base"]))
if missing_in_ours:
    print("[WARN] Missing in your CSV (expected with _adapter):", missing_in_ours)
if missing_in_base:
    print("[WARN] Missing in baseline CSV:", missing_in_base)

common = [m for m in baseline_models if m in set(base_f["model_base"]).intersection(ours_f["model_base"])]
if not common:
    raise ValueError("No overlapping model pairs after mapping. Check names and suffixes.")

base_f = base_f[base_f["model_base"].isin(common)].copy()
ours_f = ours_f[ours_f["model_base"].isin(common)].copy()

base_f["source"] = "Baseline"
ours_f["source"] = "Ours"
base_f["model_canon"] = base_f["model_base"]
ours_f["model_canon"] = ours_f["model_base"]

df = pd.concat(
    [
        base_f[["model_canon", "source", "gflops", "peak_mem_mib", "wall_time_s"]],
        ours_f[["model_canon", "source", "gflops", "peak_mem_mib", "wall_time_s"]],
    ],
    ignore_index=True,
)

model_order = [m for m in baseline_models if m in common]  # preserves your preferred order

def pivot_metric(metric: str) -> pd.DataFrame:
    p = df.pivot_table(index="model_canon", columns="source", values=metric, aggfunc="mean")
    for col in ["Baseline", "Ours"]:
        if col not in p.columns:
            p[col] = np.nan
    return p.reindex(model_order)

p_flops = pivot_metric("gflops")
p_mem   = pivot_metric("peak_mem_mib")
p_time  = pivot_metric("wall_time_s")

def maybe_log_axis(ax, values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) >= 2:
        vmax, vmin = float(np.max(vals)), float(np.min(vals))
        if vmin > 0 and vmax / vmin >= 100:
            ax.set_yscale("log")

# -------- Plot (single figure, three rows, shared x) --------
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                         figsize=(8.3*cm, 11*cm),  # <-- set height (10 cm) as needed
                         constrained_layout=True)
x = np.arange(len(model_order))
width = 0.38

def plot_grouped(ax, pivot_df, ylabel, title, show_legend=False, maybe_log=True, value_fmt="{:.2f}"):
    r1 = ax.bar(x - width/2, pivot_df["Baseline"].values, width, label="Baseline")
    r2 = ax.bar(x + width/2, pivot_df["Ours"].values, width, label="Ours")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    if show_legend:
        ax.legend(ncol=2, frameon=False)
    if maybe_log:
        all_vals = np.r_[pivot_df["Baseline"].values, pivot_df["Ours"].values]
        maybe_log_axis(ax, all_vals)
    # Compact grid + light value labels (optional; comment out if too dense)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    for rects in (r1, r2):
        for r in rects:
            h = r.get_height()
            if np.isfinite(h):
                ax.annotate(value_fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                            textcoords="offset points", xytext=(0, 3),
                            ha='center', va='bottom', fontsize=7, annotation_clip=False)
    all_vals = np.r_[pivot_df["Baseline"].values, pivot_df["Ours"].values]
    ymax = np.nanmax(all_vals)
    ax.set_ylim(top=ymax * 1.2)  # increase if still tight

plot_grouped(axes[0], p_flops, "GFLOPs", "Compute", show_legend=True, maybe_log=False, value_fmt="{:.0f}")
plot_grouped(axes[1], p_mem,   "Peak Memory (MiB)", "Memory", maybe_log=False, value_fmt="{:.0f}")
plot_grouped(axes[2], p_time,  "Wall Time (s)", "Runtime", maybe_log=False, value_fmt="{:.2f}")

# Shared x-axis labeling only on bottom
axes[-1].set_xticks(x)
plot_model_names = ['ViT-S', 'ViT-B', 'ViT-L', 'ViT-H+']
axes[-1].set_xticklabels(plot_model_names, rotation=15, ha="right")

fig.savefig("complexity_comparison_all_in_one.svg", dpi=220)
