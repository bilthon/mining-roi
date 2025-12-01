#!/usr/bin/env python3
import argparse
from pathlib import Path

from config import (
    BTC_PRICE_NOW_USD,
    CSV_PATH,
    CURTAILMENT_ELECTRICITY_USD_PER_KWH,
    CURTAILMENT_ENABLED,
    CURTAILMENT_HOURS_PER_WEEK,
    DIFF_MIN_HEIGHT,
    ELECTRICITY_USD_PER_KWH,
    REDUCED_SLOPE_FACTOR,
    RIGS_DIR,
    YEARS_HORIZON,
)
from data_loader import load_all_rigs, load_difficulty_data, load_rig_config
from difficulty_model import fit_difficulty_exp
from mining_simulator import simulate_miner
from plotting import (
    plot_difficulty_projections,
    plot_multi_rig_comparison,
    plot_daily_profit,
    plot_price_projection,
    plot_single_rig_roi,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bitcoin mining ROI simulator")
    parser.add_argument("rig_config", nargs="?", help="Path to rig config JSON")
    parser.add_argument(
        "--rigs-dir",
        default=None,
        help="Directory containing rig JSON configs (used when no rig is specified)",
    )
    parser.add_argument("--diff", action="store_true", help="Include difficulty projection plot")
    parser.add_argument("--price", action="store_true", help="Include BTC price projection plot")
    return parser.parse_args()


def print_rig_summary(name, hashrate_ths, efficiency_j_per_th, equipment_price_usd):
    print(f"Rig: {name}")
    print(f"  Hashrate:   {hashrate_ths} TH/s")
    print(f"  Efficiency: {efficiency_j_per_th} J/TH")
    print(f"  Price:      ${equipment_price_usd}")


def run_single_rig(rig_path: Path, diff_info: dict, args):
    rig = load_rig_config(rig_path)

    name = rig.get("name", "Unnamed rig")
    hashrate_ths = float(rig["hashrate_ths"])
    efficiency_j_per_th = float(rig["efficiency_j_per_th"])
    equipment_price_usd = float(rig["equipment_price_usd"])

    print_rig_summary(name, hashrate_ths, efficiency_j_per_th, equipment_price_usd)

    df_orig, equip_sats, roi_sats_orig, roi_usd_orig = simulate_miner(
        diff_info,
        slope_factor=1.0,
        hashrate_ths=hashrate_ths,
        efficiency_j_per_th=efficiency_j_per_th,
        equipment_price_usd=equipment_price_usd,
        electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
        curtailment_enabled=CURTAILMENT_ENABLED,
        curtailment_hours_per_week=CURTAILMENT_HOURS_PER_WEEK,
        curtailment_electricity_usd_per_kwh=CURTAILMENT_ELECTRICITY_USD_PER_KWH,
        btc_price_now_usd=BTC_PRICE_NOW_USD,
        years_horizon=YEARS_HORIZON,
    )

    print("\nEquipment cost (sats):", equip_sats)
    print("Final cumulative sats (orig slope):", df_orig["cumulative_sats"].iloc[-1])
    print("ROI epoch index in sats (orig slope):", roi_sats_orig)
    print("Final cumulative USD (orig slope):", df_orig["cumulative_usd"].iloc[-1])
    print("ROI epoch index in USD (orig slope):", roi_usd_orig)

    df_red, _, roi_sats_red, roi_usd_red = simulate_miner(
        diff_info,
        slope_factor=REDUCED_SLOPE_FACTOR,
        hashrate_ths=hashrate_ths,
        efficiency_j_per_th=efficiency_j_per_th,
        equipment_price_usd=equipment_price_usd,
        electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
        curtailment_enabled=CURTAILMENT_ENABLED,
        curtailment_hours_per_week=CURTAILMENT_HOURS_PER_WEEK,
        curtailment_electricity_usd_per_kwh=CURTAILMENT_ELECTRICITY_USD_PER_KWH,
        btc_price_now_usd=BTC_PRICE_NOW_USD,
        years_horizon=YEARS_HORIZON,
    )

    print("\nFinal cumulative sats (reduced slope):", df_red["cumulative_sats"].iloc[-1])
    print("ROI epoch index in sats (reduced slope):", roi_sats_red)
    print("Final cumulative USD (reduced slope):", df_red["cumulative_usd"].iloc[-1])
    print("ROI epoch index in USD (reduced slope):", roi_usd_red)

    plot_single_rig_roi(name, df_orig, df_red)

    plot_daily_profit(name, df_orig, plot_type="line")

    if args.price:
        plot_price_projection(df_orig, name)


def run_multi_rig(rigs_dir: Path, diff_info: dict, args):
    rig_entries = load_all_rigs(rigs_dir)
    if not rig_entries:
        print(f"No rig configs found in {rigs_dir}")
        return

    rig_results = []
    for entry in rig_entries:
        config = entry["config"]
        name = config.get("name", entry["path"].stem)
        hashrate_ths = float(config["hashrate_ths"])
        efficiency_j_per_th = float(config["efficiency_j_per_th"])
        equipment_price_usd = float(config["equipment_price_usd"])

        print_rig_summary(name, hashrate_ths, efficiency_j_per_th, equipment_price_usd)

        df_orig, equip_sats, roi_sats, roi_usd = simulate_miner(
            diff_info,
            slope_factor=1.0,
            hashrate_ths=hashrate_ths,
            efficiency_j_per_th=efficiency_j_per_th,
            equipment_price_usd=equipment_price_usd,
            electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
            curtailment_enabled=CURTAILMENT_ENABLED,
            curtailment_hours_per_week=CURTAILMENT_HOURS_PER_WEEK,
            curtailment_electricity_usd_per_kwh=CURTAILMENT_ELECTRICITY_USD_PER_KWH,
            btc_price_now_usd=BTC_PRICE_NOW_USD,
            years_horizon=YEARS_HORIZON,
        )

        print("  Equipment cost (sats):", equip_sats)
        print("  Final cumulative sats:", df_orig["cumulative_sats"].iloc[-1])
        print("  ROI epoch index in sats:", roi_sats)
        print("  Final cumulative USD:", df_orig["cumulative_usd"].iloc[-1])
        print("  ROI epoch index in USD:", roi_usd)
        print()

        rig_results.append({"name": name, "df": df_orig})

    plot_multi_rig_comparison(
        rig_results,
        value_key="cumulative_sats",
        ylabel="Cumulative profit (millions of sats)",
        title="All Rigs – Cumulative Sats ROI (Realistic Projection)",
    )

    plot_multi_rig_comparison(
        rig_results,
        value_key="cumulative_usd",
        ylabel="Cumulative profit (USD)",
        title="All Rigs – Cumulative USD ROI (Realistic Projection)",
    )

    if args.price and rig_results:
        plot_price_projection(rig_results[0]["df"], "All rigs")


def main():
    args = parse_args()

    rigs_dir = Path(args.rigs_dir).expanduser() if args.rigs_dir else RIGS_DIR

    df = load_difficulty_data(CSV_PATH)
    diff_info = fit_difficulty_exp(df, DIFF_MIN_HEIGHT)
    print("Fitted difficulty slope b:", diff_info["b"])

    if args.diff:
        plot_difficulty_projections(diff_info)

    if args.rig_config:
        run_single_rig(Path(args.rig_config), diff_info, args)
    else:
        run_multi_rig(rigs_dir, diff_info, args)


if __name__ == "__main__":
    main()
