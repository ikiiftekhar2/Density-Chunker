"""Professional figure generators for chunking analysis.

Each function accepts the pre-loaded data list and saves a publication-quality
figure to the figures/ directory.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .style import (
    FIGURES_DIR, BLUE, PURPLE, GREEN, RED, ORANGE, GRAY, DARK,
    METHOD_COLORS, DIM_MARKER, DIM_LABEL, color_for, label_for, marker_for, save,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

RESULT_FILES = [
    "density_legalbench_1024d.json",
    "density_legalbench_512d.json",
    "density_narrativeqa_1024d.json",
    "density_narrativeqa_512d.json",
    "legalbench_baselines_1024d.json",
    "legalbench_baselines_512d.json",
    "narrativeqa_baselines_1024d.json",
    "narrativeqa_baselines_512d.json",
]


def load_all(results_dir: Path) -> list[dict]:
    import json
    all_results = []
    for fname in RESULT_FILES:
        fpath = results_dir / fname
        if not fpath.exists():
            continue
        if "legalbench" in fname:
            dataset = "LegalBench-RAG"
        elif "narrativeqa" in fname:
            dataset = "NarrativeQA"
        else:
            dataset = "unknown"
        dim = 1024 if "1024d" in fname else 512
        with open(fpath) as fh:
            data = json.load(fh)
        for entry in data:
            entry["_dataset"] = dataset
            if "dim" not in entry:
                entry["dim"] = dim
            all_results.append(entry)
    return all_results


def _legalbench(results: list[dict]) -> list[dict]:
    return [r for r in results if r["_dataset"] == "LegalBench-RAG"]


def _narrativeqa(results: list[dict]) -> list[dict]:
    return [r for r in results if r["_dataset"] == "NarrativeQA"]


# ===========================================================================
# Fig 1 — Size Dominance Effect (HERO FIGURE)
# ===========================================================================

def fig1_size_dominance(results: list[dict]):
    """Recall@5 vs Average Chunk Size — all methods, both dims, LegalBench."""
    data = _legalbench(results)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # Plot each point
    plotted_labels = set()
    for r in data:
        method = r["method"]
        dim = r["dim"]
        x = r["intrinsic"]["avg_chunk_sentences"]
        y = r["retrieval"]["recall@5"]
        c = color_for(method)
        mk = marker_for(dim)
        ms = 110 if dim == 1024 else 85
        lbl = f"{label_for(method)} ({DIM_LABEL[dim]})"

        kw = dict(c=c, marker=mk, s=ms, edgecolors="white", linewidth=0.6, zorder=5)
        if lbl not in plotted_labels:
            ax.scatter(x, y, label=lbl, **kw)
            plotted_labels.add(lbl)
        else:
            ax.scatter(x, y, **kw)

    # Quadratic trend
    xs = [r["intrinsic"]["avg_chunk_sentences"] for r in data]
    ys = [r["retrieval"]["recall@5"] for r in data]
    z = np.polyfit(xs, ys, 2)
    x_smooth = np.linspace(min(xs) - 1, max(xs) + 2, 150)
    ax.plot(x_smooth, np.poly1d(z)(x_smooth), "--", color=GRAY, alpha=0.5, lw=1.8,
            label="Quadratic fit")

    ax.set_xlabel("Average Chunk Size (sentences)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Recall@5 vs Average Chunk Size", fontweight="bold", pad=12)
    ax.text(0.02, 0.97, "LegalBench-RAG", transform=ax.transAxes, fontsize=9,
            va="top", color=GRAY, fontstyle="italic")

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=7.5, ncol=2, loc="lower right",
              title="Method (dimension)", title_fontsize=8)
    ax.set_ylim(0.10, 0.82)

    save(fig, "fig1_size_dominance.png")
    print("  [1/9] fig1_size_dominance.png")


# ===========================================================================
# Fig 2 — Boundary Quality Isolation
# ===========================================================================

def fig2_boundary_quality(results: list[dict]):
    """Grouped bar chart: methods within same size bin."""
    data = _legalbench(results)

    bins = {
        "~5 sentences":       {"methods": ["fixed-5"]},
        "~10–12 sentences":   {"methods": ["fixed-10", "semantic-p10"]},
        "~38–40 sentences":   {"methods": ["fixed-40", "semantic-p3", "density"]},
    }

    # Aggregate
    bin_entries = defaultdict(list)
    for r in data:
        for bname, binfo in bins.items():
            if r["method"] in binfo["methods"]:
                bin_entries[bname].append(r)

    plot_order = list(bins.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for ax, bname in zip(axes, plot_order):
        entries = bin_entries[bname]
        by_method = defaultdict(list)
        for e in entries:
            by_method[e["method"]].append(e["retrieval"]["recall@5"])

        methods_sorted = sorted(by_method.keys())
        means = [np.mean(by_method[m]) for m in methods_sorted]
        stds = [np.std(by_method[m]) for m in methods_sorted]
        labels = [label_for(m) for m in methods_sorted]
        cs = [color_for(m) for m in methods_sorted]

        x = np.arange(len(methods_sorted))
        bars = ax.bar(x, means, color=cs, edgecolor="white", linewidth=0.8,
                       width=0.55, yerr=stds, capsize=4)

        # Value labels on bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=DARK)

        # Delta annotation
        if len(means) >= 2:
            delta = (max(means) - min(means)) * 100
            ax.text(0.5, 0.08, f"Δ = {delta:.1f} pp", transform=ax.transAxes,
                    ha="center", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF3CD",
                              edgecolor="#E0D5A0", alpha=0.95))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(bname, fontweight="bold", fontsize=12, pad=10)
        ax.set_ylim(0, 0.92)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        if ax is axes[0]:
            ax.set_ylabel("Recall@5")

    fig.suptitle("Boundary Quality Isolation — Matched Chunk Sizes",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.text(0.02, 0.98, "LegalBench-RAG", fontsize=9, color=GRAY, fontstyle="italic")
    fig.tight_layout()
    save(fig, "fig2_boundary_quality.png")
    print("  [2/9] fig2_boundary_quality.png")


# ===========================================================================
# Fig 3 — Dimensionality Tradeoff
# ===========================================================================

def fig3_dimensionality(results: list[dict]):
    """512d vs 1024d per method — line chart."""
    data = _legalbench(results)
    methods_order = ["fixed-5", "fixed-10", "fixed-40", "semantic-p10", "semantic-p3", "density"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method in methods_order:
        pts_512 = [r for r in data if r["method"] == method and r["dim"] == 512]
        pts_1024 = [r for r in data if r["method"] == method and r["dim"] == 1024]
        if not pts_512 or not pts_1024:
            continue

        y5 = np.mean([p["retrieval"]["recall@5"] for p in pts_512])
        y10 = np.mean([p["retrieval"]["recall@5"] for p in pts_1024])
        c = color_for(method)
        lbl = label_for(method)

        ax.plot([512, 1024], [y5, y10], "o-", color=c, lw=2.5, ms=10,
                markeredgecolor="white", markeredgewidth=0.6, label=lbl)

        # Annotate values
        delta = (y10 - y5) * 100
        ax.annotate(f"{y5:.1%}", (512, y5), textcoords="offset points",
                    xytext=(-30, 8), fontsize=8.5, color=c, fontweight="bold")
        ax.annotate(f"{y10:.1%}  (Δ{delta:+.1f}pp)", (1024, y10), textcoords="offset points",
                    xytext=(6, 8 if delta >= 0 else -14), fontsize=8.5, color=c, fontweight="bold")

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Recall@5")
    ax.set_title("Dimensionality Tradeoff — 512d vs 1024d", fontweight="bold", pad=12)
    ax.text(0.02, 0.97, "LegalBench-RAG", transform=ax.transAxes, fontsize=9,
            va="top", color=GRAY, fontstyle="italic")
    ax.set_xlim(400, 1150)
    ax.set_xticks([512, 1024])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0.40, 0.80)
    ax.legend(fontsize=9.5, loc="lower right")

    save(fig, "fig3_dimensionality.png")
    print("  [3/9] fig3_dimensionality.png")


# ===========================================================================
# Fig 4 — Efficiency Frontier (Pareto)
# ===========================================================================

def fig4_pareto_frontier(results: list[dict]):
    """Recall@5 vs Total Chunks with Pareto frontier."""
    data = _legalbench(results)

    # Aggregate by method+dim
    agg = defaultdict(list)
    for r in data:
        key = (r["method"], r["dim"])
        agg[key].append((r["total_chunks"], r["retrieval"]["recall@5"]))

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    points = []
    for (method, dim), vals in agg.items():
        avg_c = np.mean([v[0] for v in vals])
        avg_r = np.mean([v[1] for v in vals])
        points.append((avg_c, avg_r, method, dim))

    for avg_c, avg_r, method, dim in points:
        c = color_for(method)
        mk = marker_for(dim)
        ax.scatter(avg_c, avg_r, c=c, marker=mk, s=140, edgecolors="white",
                   linewidth=0.6, zorder=5,
                   label=f"{label_for(method)} ({DIM_LABEL[dim]})")

    # Pareto frontier
    pts_by_chunks = sorted(points, key=lambda p: p[0])
    pareto_x, pareto_y = [], []
    max_r = -1
    for avg_c, avg_r, method, dim in pts_by_chunks:
        if avg_r > max_r:
            pareto_x.append(avg_c)
            pareto_y.append(avg_r)
            max_r = avg_r

    # Also reverse pass
    pts_by_recall = sorted(points, key=lambda p: p[1], reverse=True)
    best_x, best_y = [], []
    min_c = float("inf")
    for avg_c, avg_r, method, dim in pts_by_recall:
        if avg_c < min_c:
            best_x.append(avg_c)
            best_y.append(avg_r)
            min_c = avg_c
    # Combine and sort
    frontier = sorted(set(zip(pareto_x + best_x, pareto_y + best_y)))
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "--", color=GRAY, alpha=0.5, lw=1.8)

        # Shade dominated region
        ax.fill_between(fx + (fx[-1] + 10000,), fy + (fy[-1],),
                        [0] * (len(fx) + 1), alpha=0.04, color=GRAY)

    ax.set_xlabel("Total Chunks (index size proxy)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Efficiency Frontier — Recall@5 vs Index Size", fontweight="bold", pad=12)
    ax.text(0.02, 0.97, "LegalBench-RAG", transform=ax.transAxes, fontsize=9,
            va="top", color=GRAY, fontstyle="italic")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(fontsize=7.5, ncol=2, title="Method (dimension)", title_fontsize=8)

    save(fig, "fig4_pareto_frontier.png")
    print("  [4/9] fig4_pareto_frontier.png")


# ===========================================================================
# Fig 5 — Intrinsic Metrics vs Retrieval Quality
# ===========================================================================

def fig5_intrinsic_vs_retrieval(results: list[dict]):
    """Four-panel: Cohesion, Separation, Size CoV, Total Chunks vs Recall@5."""
    data = _legalbench(results)

    panels = [
        ("cohesion", "Cohesion"),
        ("separation", "Separation"),
        ("size_cov", "Size Coefficient of Variation"),
        ("total_chunks", "Total Chunks"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, (key, label) in zip(axes, panels):
        plotted = set()
        for r in data:
            method = r["method"]
            dim = r["dim"]
            x = r["intrinsic"][key] if key != "total_chunks" else r["total_chunks"]
            y = r["retrieval"]["recall@5"]
            c = color_for(method)
            mk = marker_for(dim)

            lbl = f"{label_for(method)} ({DIM_LABEL[dim]})"
            if lbl not in plotted:
                ax.scatter(x, y, c=c, marker=mk, s=70, edgecolors="white",
                           linewidth=0.4, zorder=5, label=lbl)
                plotted.add(lbl)
            else:
                ax.scatter(x, y, c=c, marker=mk, s=70, edgecolors="white",
                           linewidth=0.4, zorder=5)

        ax.set_xlabel(label)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if ax is axes[0]:
            ax.set_ylabel("Recall@5")

        # Special formatting
        if key == "total_chunks":
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

    axes[0].set_title("Cohesion vs Recall@5", fontweight="bold", fontsize=11)
    axes[1].set_title("Separation vs Recall@5", fontweight="bold", fontsize=11)
    axes[2].set_title("Size CoV vs Recall@5", fontweight="bold", fontsize=11)
    axes[3].set_title("Total Chunks vs Recall@5", fontweight="bold", fontsize=11)

    fig.suptitle("Intrinsic Metrics Do Not Predict Retrieval Quality",
                 fontweight="bold", fontsize=14, y=1.03)
    fig.text(0.01, 0.98, "LegalBench-RAG", fontsize=9, color=GRAY, fontstyle="italic")

    # Compact legend shared at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    save(fig, "fig5_intrinsic_vs_retrieval.png")
    print("  [5/9] fig5_intrinsic_vs_retrieval.png")


# ===========================================================================
# Fig 6 — Recall@k Curves
# ===========================================================================

def fig6_recall_at_k(results: list[dict]):
    """Recall@k for k=1,3,5,10 — all methods + size-matched subset."""
    data = _legalbench(results)
    ks = [1, 3, 5, 10]
    all_methods = ["fixed-5", "fixed-10", "fixed-40", "semantic-p10", "semantic-p3", "density"]
    size_matched = ["fixed-40", "semantic-p3", "density"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: all methods
    for method in all_methods:
        for dim in [512, 1024]:
            entries = [r for r in data if r["method"] == method and r["dim"] == dim]
            if not entries:
                continue
            r = entries[0]
            recall_vals = [r["retrieval"][f"recall@{k}"] for k in ks]
            c = color_for(method)
            ls = "-" if dim == 1024 else "--"
            alpha = 0.95 if dim == 1024 else 0.65
            ax1.plot(ks, recall_vals, color=c, linestyle=ls, marker=marker_for(dim),
                     ms=8, alpha=alpha, lw=2.2,
                     label=f"{label_for(method)} ({DIM_LABEL[dim]})")

    ax1.set_xlabel("k")
    ax1.set_ylabel("Recall@k")
    ax1.set_xticks(ks)
    ax1.set_title("All Methods", fontweight="bold", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(fontsize=7, ncol=2)

    # Panel B: size-matched only
    for method in size_matched:
        for dim in [512, 1024]:
            entries = [r for r in data if r["method"] == method and r["dim"] == dim]
            if not entries:
                continue
            r = entries[0]
            recall_vals = [r["retrieval"][f"recall@{k}"] for k in ks]
            c = color_for(method)
            ls = "-" if dim == 1024 else "--"
            alpha = 0.95 if dim == 1024 else 0.65
            ax2.plot(ks, recall_vals, color=c, linestyle=ls, marker=marker_for(dim),
                     ms=10, alpha=alpha, lw=2.8,
                     label=f"{label_for(method)} ({DIM_LABEL[dim]})")

    ax2.set_xlabel("k")
    ax2.set_ylabel("Recall@k")
    ax2.set_xticks(ks)
    ax2.set_title("Size-Matched (~38–40 sent/chunk)", fontweight="bold", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=9)

    fig.suptitle("Recall@k Curves", fontweight="bold", fontsize=14, y=1.02)
    fig.text(0.01, 0.98, "LegalBench-RAG", fontsize=9, color=GRAY, fontstyle="italic")
    fig.tight_layout()
    save(fig, "fig6_recall_at_k.png")
    print("  [6/9] fig6_recall_at_k.png")


# ===========================================================================
# Fig 7 — NarrativeQA Diagnosis
# ===========================================================================

def fig7_narrativeqa_diagnosis(results: list[dict]):
    """Bar chart of NarrativeQA ROUGE-L + diagnosis annotations."""
    data = _narrativeqa(results)

    # Aggregate
    by_method = defaultdict(list)
    for r in data:
        key = (r["method"], r["dim"])
        by_method[key].append(r.get("rouge_l", 0))

    keys_sorted = sorted(by_method.keys(), key=lambda kd: np.mean(by_method[kd]), reverse=True)
    labels = [f"{label_for(m)}\n({DIM_LABEL[d]})" for m, d in keys_sorted]
    means = [np.mean(by_method[k]) for k in keys_sorted]
    colors = [color_for(m) for m, d in keys_sorted]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(keys_sorted))

    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ROUGE-L")
    ax.set_title("NarrativeQA — All Methods Near Random", fontweight="bold", pad=12)

    # Diagnosis box
    nq_chunks_avg = np.mean([r.get("total_chunks", 0) for r in data])
    random_chance = 10 / nq_chunks_avg * 100 if nq_chunks_avg > 0 else 0
    diag_text = (
        f"All ROUGE-L < 1.5%\n"
        f"~{nq_chunks_avg:,.0f} chunks per document\n"
        f"Random baseline (top-10 from {nq_chunks_avg:,.0f}): {random_chance:.2f}%\n"
        f"Nightmare retrieval scenario:\n"
        f"needle in a haystack"
    )
    ax.text(0.98, 0.96, diag_text, transform=ax.transAxes, fontsize=9.5,
            va="top", ha="right", fontstyle="italic", color=DARK,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#FADBD8", edgecolor="#E6B0AA",
                      alpha=0.9))

    ax.axhline(y=0.015, color=RED, linestyle=":", alpha=0.6, lw=1.5)
    ax.text(0.02, 0.015, "  ~1.5% (effectively random)", fontsize=8, color=RED,
            va="bottom", fontstyle="italic")

    save(fig, "fig7_narrativeqa_diagnosis.png")
    print("  [7/9] fig7_narrativeqa_diagnosis.png")


# ===========================================================================
# Fig 8 — Chunk Size Distribution
# ===========================================================================

def fig8_chunk_size_distribution(results: list[dict]):
    """Mean chunk size + CoV for variable-size methods."""
    data = _legalbench(results)
    var_methods = ["semantic-p10", "semantic-p3", "semantic-p5", "density", "recursive-512-100"]
    all_dims = [512, 1024]

    entries = []
    for method in var_methods:
        for dim in all_dims:
            for r in data:
                if r["method"] == method and r["dim"] == dim:
                    entries.append(r)

    methods_order = sorted(set(e["method"] for e in entries))
    x = np.arange(len(methods_order))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, dim in enumerate(all_dims):
        dim_by_method = {}
        for e in entries:
            if e["dim"] == dim:
                dim_by_method[e["method"]] = e

        means = [dim_by_method.get(m, {}).get("intrinsic", {}).get("avg_chunk_sentences", 0)
                 for m in methods_order]
        covs = [dim_by_method.get(m, {}).get("intrinsic", {}).get("size_cov", 0)
                for m in methods_order]
        stds = [m * c for m, c in zip(means, covs)]
        off = (i - 0.5) * width

        bars = ax.bar(x + off, means, width, yerr=stds, capsize=4,
                      color="#5DADE2" if dim == 512 else "#2E86C1",
                      alpha=0.85, edgecolor="white", linewidth=0.6,
                      label=f"{DIM_LABEL[dim]}")

        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{mean_val:.1f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in methods_order], fontsize=10)
    ax.set_ylabel("Average Chunk Size (sentences)")
    ax.set_title("Chunk Size Distribution — Mean ± CoV", fontweight="bold", pad=12)
    ax.text(0.02, 0.97, "LegalBench-RAG", transform=ax.transAxes, fontsize=9,
            va="top", color=GRAY, fontstyle="italic")

    # Cap line
    ax.axhline(y=40, color=RED, linestyle="--", alpha=0.5, lw=1.3)
    ax.text(len(methods_order) - 0.45, 41.5, "max_sentences = 40", fontsize=8.5,
            color=RED, fontstyle="italic")

    ax.legend(fontsize=9)

    save(fig, "fig8_chunk_size_distribution.png")
    print("  [8/9] fig8_chunk_size_distribution.png")


# ===========================================================================
# Fig 9 — Sigma Sensitivity
# ===========================================================================

def fig9_sigma_sensitivity(results: list[dict]):
    """Sigma values from best Optuna trials + contextual annotation."""
    sigma_entries = [r for r in results if "sigma" in r]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for s in sigma_entries:
        ds = s["_dataset"]
        dim = s["dim"]
        sigma = s["sigma"]
        avg_size = s["intrinsic"]["avg_chunk_sentences"]

        if ds == "LegalBench-RAG":
            y = s["retrieval"]["recall@5"]
            c = GREEN
            lbl = f"{ds} ({DIM_LABEL[dim]}) — Recall@5"
            ax.scatter(sigma, y, c=c, marker=marker_for(dim), s=220,
                       edgecolors="white", linewidth=1, zorder=5)
            ax.annotate(f"σ={sigma:.1f}\nR@5={y:.1%}\nsize={avg_size:.0f}",
                        (sigma, y), textcoords="offset points", xytext=(12, 12),
                        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="#ddd", alpha=0.85))
        else:
            y = avg_size
            c = ORANGE
            lbl = f"NarrativeQA ({DIM_LABEL[dim]}) — Avg Size"
            ax.scatter(sigma, y, c=c, marker=marker_for(dim), s=220,
                       edgecolors="white", linewidth=1, zorder=5)
            ax.annotate(f"σ={sigma:.1f}\nsize={avg_size:.0f} sent",
                        (sigma, y), textcoords="offset points", xytext=(12, 10),
                        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="#ddd", alpha=0.85))

    # Tweak: add fake lines so legend entries are clean
    ax.scatter([], [], c=GREEN, marker="o", s=80, label="LegalBench-RAG (Recall@5)")
    ax.scatter([], [], c=ORANGE, marker="o", s=80, label="NarrativeQA (Avg Size)")

    ax.set_xlabel("σ")
    ax.set_ylabel("Recall@5 / Average Chunk Size (sentences)")
    ax.set_title("DensityChunker σ — Best Trial Values", fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")

    # Annotation about what sigma means
    note = (
        "LegalBench σ=37–48 → Gaussian window spans nearly\nthe entire document."
        " The density signal is almost\nglobal — the max_sentences=40 constraint does\nthe splitting."
        " Position weighting is barely active."
    )
    ax.text(0.98, 0.55, note, transform=ax.transAxes, fontsize=9,
            va="center", ha="right", fontstyle="italic", color=DARK,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#EBF5FB", edgecolor="#AED6F1",
                      alpha=0.9))

    save(fig, "fig9_sigma_sensitivity.png")
    print("  [9/9] fig9_sigma_sensitivity.png")


# ===========================================================================
# Generate all
# ===========================================================================

def generate_all(results_dir: Path | None = None):
    """Generate all figures from result JSONs."""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "results" / "main"

    print(f"Loading results from: {results_dir}")
    results = load_all(results_dir)
    print(f"Loaded {len(results)} entries. Generating figures...\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig1_size_dominance(results)
    fig2_boundary_quality(results)
    fig3_dimensionality(results)
    fig4_pareto_frontier(results)
    fig5_intrinsic_vs_retrieval(results)
    fig6_recall_at_k(results)
    fig7_narrativeqa_diagnosis(results)
    fig8_chunk_size_distribution(results)
    fig9_sigma_sensitivity(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    for p in sorted(FIGURES_DIR.glob("*.png")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}  ({size_kb:.0f} KB)")
