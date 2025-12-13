#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    BTC_PRICE_NOW_USD,
    CSV_PATH,
    CURTAILMENT_ELECTRICITY_USD_PER_KWH,
    CURTAILMENT_ENABLED,
    CURTAILMENT_HOURS_PER_WEEK,
    DEFAULT_MC_SIMULATIONS,
    DIFF_MIN_HEIGHT,
    ELECTRICITY_USD_PER_KWH,
    MC_DEFAULT_BANDS,
    MC_DEFAULT_SEED,
    RIGS_DIR,
    YEARS_HORIZON,
)
from src.data_loader import load_all_rigs, load_difficulty_data, load_rig_config
from src.difficulty_model import fit_difficulty_exp
from src.monte_carlo import run_monte_carlo_for_rig, run_monte_carlo_multi_rig
from src.plotting import (
    plot_price_projection,
    plot_roi_cloud,
    plot_multi_rig_mc_comparison,
    plot_difficulty_mc_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bitcoin mining ROI simulator")
    parser.add_argument("rig_config", nargs="?", help="Path to rig config JSON")
    parser.add_argument(
        "--rigs-dir",
        default=None,
        help="Directory containing rig JSON configs (used when no rig is specified)",
    )
    parser.add_argument("--diff", action="store_true", help="Include difficulty path overlays")
    parser.add_argument("--price", action="store_true", help="Include BTC price projection plot")
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Number of Monte Carlo simulations (default from config)",
    )
    parser.add_argument(
        "--mc-seed",
        type=int,
        default=None,
        help="Optional seed for Monte Carlo residual sampling",
    )
    parser.add_argument(
        "--mc-show-paths",
        type=int,
        default=0,
        help="Number of individual Monte Carlo paths to overlay (0 = none)",
    )
    parser.add_argument(
        "--mc-bands",
        type=str,
        default=None,
        help='Comma-separated percentile bands for ROI cloud (e.g., "10-90,25-75")',
    )
    parser.add_argument(
        "--mc-show-difficulty",
        type=int,
        default=0,
        help="Number of Monte Carlo difficulty paths to overlay on historical difficulty",
    )
    return parser.parse_args()


def print_rig_summary(name, hashrate_ths, efficiency_j_per_th, equipment_price_usd):
    print(f"Rig: {name}")
    print(f"  Hashrate:   {hashrate_ths} TH/s")
    print(f"  Efficiency: {efficiency_j_per_th} J/TH")
    print(f"  Price:      ${equipment_price_usd}")


def parse_mc_bands(bands_str: Optional[str]):
    if not bands_str:
        return None
    bands = []
    for part in bands_str.split(","):
        if "-" not in part:
            continue
        lo_str, hi_str = part.split("-", 1)
        try:
            lo = float(lo_str)
            hi = float(hi_str)
        except ValueError:
            continue
        if 0 <= lo < hi <= 100:
            bands.append((lo, hi))
    return bands if bands else None


def _percentile_label(stats: dict) -> str:
    return f"P10={stats['p10']:.0f}, P50={stats['p50']:.0f}, P90={stats['p90']:.0f}"


def _parse_effective_bands(args_bands: Optional[str]):
    parsed = parse_mc_bands(args_bands) if args_bands else None
    if parsed:
        return parsed
    if MC_DEFAULT_BANDS:
        return parse_mc_bands(MC_DEFAULT_BANDS)
    return None


def _resolve_n_sims(arg_value: Optional[int]) -> int:
    if arg_value and arg_value > 0:
        return arg_value
    return DEFAULT_MC_SIMULATIONS


def _resolve_seed(arg_value: Optional[int]) -> Optional[int]:
    if arg_value is not None:
        return arg_value
    return MC_DEFAULT_SEED


def run_single_rig_mc(rig_path: Path, diff_info: dict, args):
    rig = load_rig_config(rig_path)

    name = rig.get("name", rig_path.stem)
    hashrate_ths = float(rig["hashrate_ths"])
    efficiency_j_per_th = float(rig["efficiency_j_per_th"])
    equipment_price_usd = float(rig["equipment_price_usd"])

    print_rig_summary(name, hashrate_ths, efficiency_j_per_th, equipment_price_usd)

    n_sims = _resolve_n_sims(args.n_sims)
    seed = _resolve_seed(args.mc_seed)
    bands = _parse_effective_bands(args.mc_bands)

    print(f"\nRunning Monte Carlo (random-walk) simulations ({n_sims}) for {name} ...")
    mc_result = run_monte_carlo_for_rig(
        diff_info=diff_info,
        rig=rig,
        years_horizon=YEARS_HORIZON,
        n_sims=n_sims,
        slope_factor=1.0,
        seed=seed,
        btc_price_now_usd=BTC_PRICE_NOW_USD,
        electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
        curtailment_enabled=CURTAILMENT_ENABLED,
        curtailment_hours_per_week=CURTAILMENT_HOURS_PER_WEEK,
        curtailment_electricity_usd_per_kwh=CURTAILMENT_ELECTRICITY_USD_PER_KWH,
    )

    summary = mc_result["summary"]
    sats_stats = summary["final_sats"]
    usd_stats = summary["final_usd"]

    print("Monte Carlo cumulative sats:", _percentile_label(sats_stats))
    print("Monte Carlo cumulative USD: ", _percentile_label(usd_stats))

    roi_epochs = summary["roi_sats_epochs"]
    if roi_epochs:
        roi_arr = np.asarray(roi_epochs, dtype=float)
        print(
            "ROI epoch (sats) percentiles:",
            f"P10={np.percentile(roi_arr,10):.0f},",
            f"P50={np.percentile(roi_arr,50):.0f},",
            f"P90={np.percentile(roi_arr,90):.0f}",
        )
    else:
        print("ROI epoch (sats): no simulations reached ROI within horizon (random-walk).")

    plot_roi_cloud(
        mc_result["sim_results"],
        show_paths=max(0, int(args.mc_show_paths)),
        bands=bands,
        name=name,
    )

    max_paths = max(0, int(args.mc_show_difficulty))
    if args.diff and max_paths == 0:
        max_paths = 5

    if max_paths > 0:
        plot_difficulty_mc_paths(
            diff_info["df_fit"],
            mc_result["difficulty_paths"],
            mc_result["timeline"]["heights"],
            max_paths=max_paths,
            title=f"{name} – Historical vs Monte Carlo Difficulty (Random-Walk)",
        )

    if args.price:
        # Use the first simulation path as a representative trajectory for the price projection plot
        plot_price_projection(mc_result["sim_results"][0]["df"], name)


def run_multi_rig_mc(rigs_dir: Path, diff_info: dict, args):
    rig_entries = load_all_rigs(rigs_dir)
    if not rig_entries:
        print(f"No rig configs found in {rigs_dir}")
        return

    n_sims = _resolve_n_sims(args.n_sims)
    seed = _resolve_seed(args.mc_seed)
    bands = _parse_effective_bands(args.mc_bands)

    print(f"\nRunning Monte Carlo (random-walk) simulations ({n_sims}) for {len(rig_entries)} rigs ...")
    mc_results = run_monte_carlo_multi_rig(
        diff_info=diff_info,
        rig_entries=rig_entries,
        years_horizon=YEARS_HORIZON,
        n_sims=n_sims,
        slope_factor=1.0,
        seed=seed,
        btc_price_now_usd=BTC_PRICE_NOW_USD,
        electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
        curtailment_enabled=CURTAILMENT_ENABLED,
        curtailment_hours_per_week=CURTAILMENT_HOURS_PER_WEEK,
        curtailment_electricity_usd_per_kwh=CURTAILMENT_ELECTRICITY_USD_PER_KWH,
    )

    for entry in rig_entries:
        rig_cfg = entry["config"]
        name = rig_cfg.get("name", entry["path"].stem)
        hashrate_ths = float(rig_cfg["hashrate_ths"])
        efficiency_j_per_th = float(rig_cfg["efficiency_j_per_th"])
        equipment_price_usd = float(rig_cfg["equipment_price_usd"])
        print_rig_summary(name, hashrate_ths, efficiency_j_per_th, equipment_price_usd)

        summary = mc_results[name]["summary"]
        print("  Monte Carlo cumulative sats:", _percentile_label(summary["final_sats"]))
        print("  Monte Carlo cumulative USD :", _percentile_label(summary["final_usd"]))
        roi_epochs = summary["roi_sats_epochs"]
        if roi_epochs:
            roi_arr = np.asarray(roi_epochs, dtype=float)
            print(
                "  ROI epoch (sats) percentiles:",
                f"P10={np.percentile(roi_arr,10):.0f},",
                f"P50={np.percentile(roi_arr,50):.0f},",
                f"P90={np.percentile(roi_arr,90):.0f}",
            )
        else:
            print("  ROI epoch (sats): no simulations reached ROI within horizon.")
        print()

    plot_multi_rig_mc_comparison(mc_results, bands=bands)

    max_paths = max(0, int(args.mc_show_difficulty))
    if args.diff and max_paths == 0:
        max_paths = 5

    if max_paths > 0:
        first_result = next(iter(mc_results.values()))
        rig_names = list(mc_results.keys())
        if len(rig_names) <= 4:
            names_label = ", ".join(rig_names)
        else:
            names_label = ", ".join(rig_names[:3]) + f", +{len(rig_names) - 3} more"

        plot_difficulty_mc_paths(
            diff_info["df_fit"],
            first_result["difficulty_paths"],
            first_result["timeline"]["heights"],
            max_paths=max_paths,
            title=f"Historical vs Monte Carlo Difficulty (Random-Walk) – {names_label}",
        )

    if args.price:
        # Use the first rig's first simulation for price projection visualization
        first_result = next(iter(mc_results.values()))
        plot_price_projection(first_result["sim_results"][0]["df"], "Multi-rig (sample path)")


def main():
    args = parse_args()

    rigs_dir = Path(args.rigs_dir).expanduser() if args.rigs_dir else RIGS_DIR

    df = load_difficulty_data(CSV_PATH)
    diff_info = fit_difficulty_exp(df, DIFF_MIN_HEIGHT)
    print("Fitted difficulty slope b:", diff_info["b"])

    if args.rig_config:
        run_single_rig_mc(Path(args.rig_config), diff_info, args)
    else:
        run_multi_rig_mc(rigs_dir, diff_info, args)


if __name__ == "__main__":
    main()
