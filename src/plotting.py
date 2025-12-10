from datetime import datetime, timezone
from typing import Iterable, Literal, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .config import REDUCED_SLOPE_FACTOR, YEARS_HORIZON


def plot_difficulty_projections(diff_info: dict) -> None:
    df_700k = diff_info["df_fit"]
    t0 = diff_info["t0"]
    h0 = diff_info["h0"]
    D0 = diff_info["D0"]
    b_orig = diff_info["b"]

    def D_orig(t):
        return D0 * np.exp(b_orig * (t - t0))

    def D_reduced(t):
        return D0 * np.exp(b_orig * REDUCED_SLOPE_FACTOR * (t - t0))

    blocks_per_epoch = 2016
    epoch_seconds = blocks_per_epoch * 600
    seconds_per_year = 365 * 24 * 3600
    total_seconds = YEARS_HORIZON * seconds_per_year
    n_epochs = int(total_seconds // epoch_seconds) + 1

    epoch_idx = np.arange(n_epochs)
    t_future = t0 + epoch_idx * epoch_seconds
    h_future = h0 + epoch_idx * blocks_per_epoch

    D_future_orig = D_orig(t_future)
    D_future_red = D_reduced(t_future)

    def difficulty_formatter(value, _):
        if value >= 1e18:
            return f"{value / 1e18:.2f}H"
        if value >= 1e15:
            return f"{value / 1e15:.2f}P"
        if value >= 1e12:
            return f"{value / 1e12:.2f}T"
        return f"{value:.0f}"

    def height_to_date_num(height):
        timestamp = t0 + (height - h0) * 600
        if np.isscalar(timestamp):
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return mdates.date2num(dt)
        dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamp]
        return mdates.date2num(dates)

    def date_num_to_height(date_num):
        def _ensure_datetime(dt_obj):
            while isinstance(dt_obj, (list, tuple, np.ndarray)):
                if len(dt_obj) == 0:
                    raise ValueError("Received empty datetime container")
                dt_obj = dt_obj[0]
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj

        date_arr = np.asarray(date_num)
        was_scalar = date_arr.ndim == 0

        dts = np.asarray(mdates.num2date(date_arr), dtype=object)
        timestamps = np.vectorize(lambda dt: _ensure_datetime(dt).timestamp())(dts)
        heights = h0 + (timestamps - t0) / 600

        if was_scalar:
            return heights.item()
        return heights

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)

    ax1.step(
        df_700k["height"],
        df_700k["difficulty"],
        where="post",
        label="Real difficulty (≥700k)",
    )
    ax1.step(
        h_future,
        D_future_orig,
        where="post",
        color="#6aaed6",
        linewidth=1.6,
        label="Projection: original slope",
    )
    ax1.step(
        h_future,
        D_future_red,
        where="post",
        color="#f4a261",
        linewidth=1.6,
        label=f"Projection: reduced slope ({REDUCED_SLOPE_FACTOR:.2f}×)",
    )
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(difficulty_formatter))
    ax1.set_ylabel("Difficulty (log scale)")
    ax1.set_xlabel("Block height")
    ax1.set_title("Log Scale")
    ax1.legend()
    ax1.grid(True, **light_grid)

    ax1_top = ax1.secondary_xaxis("top", functions=(height_to_date_num, date_num_to_height))
    ax1_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1_top.set_xlabel("Date")
    ax1_top.xaxis.set_major_locator(mdates.YearLocator())
    ax1_top.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    ax1_top.tick_params(labelsize=8)

    ax2.step(
        df_700k["height"],
        df_700k["difficulty"],
        where="post",
        label="Real difficulty (≥700k)",
    )
    ax2.step(
        h_future,
        D_future_orig,
        where="post",
        color="#6aaed6",
        linewidth=1.6,
        label="Projection: original slope",
    )
    ax2.step(
        h_future,
        D_future_red,
        where="post",
        color="#f4a261",
        linewidth=1.6,
        label=f"Projection: reduced slope ({REDUCED_SLOPE_FACTOR:.2f}×)",
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(difficulty_formatter))
    ax2.set_ylabel("Difficulty (linear scale)")
    ax2.set_xlabel("Block height")
    ax2.set_title("Linear Scale")
    ax2.legend()
    ax2.grid(True, **light_grid)

    ax2_top = ax2.secondary_xaxis("top", functions=(height_to_date_num, date_num_to_height))
    ax2_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2_top.set_xlabel("Date")
    ax2_top.xaxis.set_major_locator(mdates.YearLocator())
    ax2_top.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    ax2_top.tick_params(labelsize=8)

    fig.suptitle("Bitcoin Difficulty: Real Data Since 700k vs Projections", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_single_rig_roi(name: str, df_orig, df_red) -> None:
    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)
    plt.figure(figsize=(12, 6))
    plt.plot(df_orig["date"], df_orig["cumulative_sats"], label="Cumulative sats – original slope")
    plt.plot(
        df_red["date"],
        df_red["cumulative_sats"],
        linestyle="--",
        label=f"Cumulative sats – reduced slope ({REDUCED_SLOPE_FACTOR:.2f}×)",
    )
    plt.axhline(0, linestyle=":", label="Break-even (0 sats)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative profit (millions of sats)")
    plt.title(f"{name} – Cumulative Mining Profit (Two Difficulty Scenarios)")
    plt.legend()
    plt.grid(True, **light_grid)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df_orig["date"], df_orig["cumulative_usd"], label="Cumulative USD – original slope")
    plt.plot(
        df_red["date"],
        df_red["cumulative_usd"],
        linestyle="--",
        label=f"Cumulative USD – reduced slope ({REDUCED_SLOPE_FACTOR:.2f}×)",
    )
    plt.axhline(0, linestyle=":", label="Break-even (0 USD)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative profit (USD)")
    plt.title(f"{name} – Cumulative Mining Profit (Two Difficulty Scenarios, USD)")
    plt.legend()
    plt.grid(True, **light_grid)
    plt.tight_layout()
    plt.show()


def plot_multi_rig_comparison(rig_results: Iterable[dict], value_key: str, ylabel: str, title: str) -> None:
    light_grid = dict(which="both", color="#d0d0d0", linewidth=0.4, alpha=0.4)
    plt.figure(figsize=(12, 6))
    for entry in rig_results:
        df = entry["df"]
        plt.plot(df["date"], df[value_key], label=entry["name"])
    plt.axhline(0, linestyle=":", color="#666666", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, **light_grid)
    plt.tight_layout()
    plt.show()


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
    plt.title("Monte Carlo (random-walk) ROI – Cumulative Sats Bands")
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
        plt.title(f"Monte Carlo (random-walk) ROI – Sample Paths (n={n_paths})")
        plt.grid(True, **light_grid)
        plt.tight_layout()
        plt.show()

    roi_epochs = [entry.get("roi_sats_epoch") for entry in sim_results if entry.get("roi_sats_epoch") is not None]
    if roi_epochs:
        plt.figure(figsize=(10, 4))
        plt.hist(roi_epochs, bins=20, color="#1f77b4", alpha=0.75, edgecolor="#1f3f5b")
        plt.xlabel("Epoch index for ROI (sats)")
        plt.ylabel("Frequency")
        plt.title("Distribution of ROI Epochs (Random-Walk, Cumulative Sats >= 0)")
        plt.grid(True, axis="y", **light_grid)
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

