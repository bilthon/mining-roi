from typing import Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_price_projection(df_orig, name: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df_orig["date"], df_orig["btc_price"])
    plt.xlabel("Date")
    plt.ylabel("BTC price (USD)")
    plt.title(f"Projected BTC Price – Same Horizon as ROI Sim ({name})")
    plt.grid(True, which="both", axis="y")
    plt.tight_layout()
    plt.show()


def plot_roi_cloud(
    sim_results,
    show_paths: int = 0,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
    name: Optional[str] = None,
) -> None:
    """
    Visualize Monte Carlo cumulative ROI bands and ROI epoch distribution.
    sim_results: list of dicts containing a `df` DataFrame and `roi_sats_epoch`.
    """
    if not sim_results:
        print("No Monte Carlo results to plot.")
        return

    if bands is None or len(bands) == 0:
        bands = [(10, 90)]

    # Ensure bands are well-formed and sorted by width (outer to inner)
    cleaned = []
    for lo, hi in bands:
        lo_f, hi_f = float(lo), float(hi)
        if not (0 <= lo_f < hi_f <= 100):
            continue
        cleaned.append((lo_f, hi_f))
    if not cleaned:
        cleaned = [(10.0, 90.0)]
    bands_sorted = sorted(cleaned, key=lambda x: x[1] - x[0], reverse=True)

    dates = sim_results[0]["df"]["date"]
    cumulative_stack = np.vstack([entry["df"]["cumulative_sats"].values for entry in sim_results])

    percentiles = {p: np.percentile(cumulative_stack, p, axis=0) for p in set([50] + [p for band in bands_sorted for p in band])}
    p50 = percentiles[50]

    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)
    plt.figure(figsize=(12, 6))
    base_color = "#1f77b4"
    for idx, (lo, hi) in enumerate(bands_sorted):
        plo = percentiles[lo]
        phi = percentiles[hi]
        alpha = 0.18 + 0.10 * max(0, len(bands_sorted) - idx)
        plt.fill_between(dates, plo, phi, color="#c6dfee", alpha=min(alpha, 0.6), label=f"P{int(lo)}–P{int(hi)} cumulative sats")
    plt.plot(dates, p50, color=base_color, linewidth=1.6, label="Median cumulative sats")
    plt.axhline(0, linestyle=":", color="#666666", linewidth=0.8, label="Break-even (0 sats)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative profit (sats)")
    title_prefix = f"{name} – " if name else ""
    plt.title(f"{title_prefix}Monte Carlo (random-walk) ROI – Cumulative Sats Bands")
    plt.legend()
    plt.grid(True, **light_grid)
    plt.tight_layout()
    plt.show()

    if show_paths > 0:
        n_paths = min(show_paths, len(sim_results))
        plt.figure(figsize=(12, 6))
        for entry in sim_results[:n_paths]:
            plt.plot(entry["df"]["date"], entry["df"]["cumulative_sats"], linewidth=0.9, alpha=0.8, color="#1f77b4")
        plt.axhline(0, linestyle=":", color="#666666", linewidth=0.8, label="Break-even (0 sats)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative profit (sats)")
        plt.title(f"{title_prefix}Monte Carlo (random-walk) ROI – Sample Paths (n={n_paths})")
        plt.grid(True, **light_grid)
        plt.tight_layout()
        plt.show()

    roi_epochs = [entry.get("roi_sats_epoch") for entry in sim_results if entry.get("roi_sats_epoch") is not None]
    if roi_epochs:
        plt.figure(figsize=(10, 4))
        plt.hist(roi_epochs, bins=20, color="#1f77b4", alpha=0.75, edgecolor="#1f3f5b")
        plt.xlabel("Epoch index for ROI (sats)")
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix}Distribution of ROI Epochs (Random-Walk, Cumulative Sats >= 0)")
        plt.grid(True, axis="y", **light_grid)
        plt.tight_layout()
        plt.show()


def plot_multi_rig_mc_comparison(
    multi_rig_results: dict,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """
    Compare Monte Carlo cumulative sats across multiple rigs.
    Expects a mapping of rig name -> single-rig Monte Carlo result.
    """
    if not multi_rig_results:
        print("No multi-rig Monte Carlo results to plot.")
        return

    if bands is None or len(bands) == 0:
        bands = [(10, 90)]

    cleaned = []
    for lo, hi in bands:
        lo_f, hi_f = float(lo), float(hi)
        if not (0 <= lo_f < hi_f <= 100):
            continue
        cleaned.append((lo_f, hi_f))
    if not cleaned:
        cleaned = [(10.0, 90.0)]
    bands_sorted = sorted(cleaned, key=lambda x: x[1] - x[0], reverse=True)

    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)
    plt.figure(figsize=(12, 6))

    for rig_name, result in multi_rig_results.items():
        sim_results = result.get("sim_results", [])
        if not sim_results:
            continue
        dates = sim_results[0]["df"]["date"]
        cumulative_stack = np.vstack([entry["df"]["cumulative_sats"].values for entry in sim_results])
        percentiles = {p: np.percentile(cumulative_stack, p, axis=0) for p in set([50] + [p for band in bands_sorted for p in band])}
        p50 = percentiles[50]

        base_color = plt.gca()._get_lines.get_next_color()
        for idx, (lo, hi) in enumerate(bands_sorted):
            plo = percentiles[lo]
            phi = percentiles[hi]
            alpha = 0.12 + 0.08 * max(0, len(bands_sorted) - idx)
            plt.fill_between(
                dates,
                plo,
                phi,
                color=base_color,
                alpha=min(alpha, 0.45),
                label=f"{rig_name} P{int(lo)}–P{int(hi)}" if idx == 0 else None,
            )

        plt.plot(dates, p50, color=base_color, linewidth=1.8, label=f"{rig_name} median")

    plt.axhline(0, linestyle=":", color="#666666", linewidth=0.8, label="Break-even (0 sats)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative profit (sats)")

    rig_names = list(multi_rig_results.keys())
    if len(rig_names) <= 4:
        names_label = ", ".join(rig_names)
    else:
        names_label = ", ".join(rig_names[:3]) + f", +{len(rig_names) - 3} more"

    plt.title(f"Monte Carlo ROI – Multi-Rig Comparison ({names_label})")
    plt.legend()
    plt.grid(True, **light_grid)
    plt.tight_layout()
    plt.show()


def plot_difficulty_mc_paths(
    df_hist,
    mc_paths: np.ndarray,
    heights: np.ndarray,
    max_paths: int = 0,
    title: str = "Monte Carlo Difficulty Paths (Random-Walk)",
) -> None:
    """
    Step-plot historical difficulty and overlay sample Monte Carlo difficulty trajectories.
    mc_paths shape: (n_sims, n_epochs)
    heights shape: (n_epochs,)
    """
    if mc_paths.size == 0 or heights.size == 0:
        print("No Monte Carlo difficulty paths to plot.")
        return

    n_paths = min(max_paths, mc_paths.shape[0]) if max_paths and max_paths > 0 else 0

    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)
    plt.figure(figsize=(12, 6))

    # Historical
    plt.step(df_hist["height"], df_hist["difficulty"], where="post", label="Historical difficulty (fit window)")

    # Sample MC paths
    for path_idx in range(n_paths):
        plt.step(
            heights,
            mc_paths[path_idx],
            where="post",
            linewidth=0.9,
            alpha=0.55,
            label="MC difficulty path" if path_idx == 0 else None,
        )

    plt.yscale("log")
    plt.xlabel("Block height")
    plt.ylabel("Difficulty (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, **light_grid)
    plt.tight_layout()
    plt.show()


def plot_daily_profit(
    name: str,
    df,
    plot_type: Literal["line", "bar"] = "line",
) -> None:
    dates = df["date"]
    sats = df["daily_net_sats"]
    usd = df["daily_net_usd"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if plot_type == "bar":
        ax1.bar(dates, sats, width=8.0, color="#ff7f0e")
        ax2.bar(dates, usd, width=8.0, color="#2a851c")
    else:
        ax1.plot(dates, sats, label="Daily net sats", color="#ff7f0e")
        ax2.plot(dates, usd, label="Daily net USD", color="#2a851c")

    ax1.axhline(0, linestyle=":", color="#666666", linewidth=0.8)
    ax2.axhline(0, linestyle=":", color="#666666", linewidth=0.8)

    ax1.set_ylabel("Net sats / day")
    ax2.set_ylabel("Net USD / day")
    ax2.set_xlabel("Date")

    ax1.set_title(f"{name} – Daily Mining Profit (Realistic Difficulty)")

    if plot_type == "line":
        ax1.legend()
        ax2.legend()

    ax1.grid(True, which="both", axis="y")
    ax2.grid(True, which="both", axis="y")

    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)
    ax1.grid(True, **light_grid)
    ax2.grid(True, **light_grid)

    plt.tight_layout()
    plt.show()

