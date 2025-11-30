#!/usr/bin/env python3
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone

# ------------ GLOBAL CONFIG (non-rig stuff) -----------------

CSV_PATH = "difficulty_epochs.csv"

ELECTRICITY_USD_PER_KWH = 0.05
BTC_PRICE_NOW_USD = 90_000.0

FEE_SATS_PER_BLOCK = 1_000_000
YEARS_HORIZON = 4
DIFF_MIN_HEIGHT = 700_000

# Reduced slope factor for the "slower difficulty" scenario
REDUCED_SLOPE_FACTOR = 0.75

# Porkopolis-like power law params (price vs days since genesis)
PL_A = 1.44e-17
PL_B = 5.78
GENESIS = datetime(2009, 1, 3, tzinfo=timezone.utc)

SATS_PER_BTC = 100_000_000

# ------------------------------------------------------------


def load_rig_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def fit_difficulty_exp(df: pd.DataFrame, min_height: int):
    df2 = df[df["height"] >= min_height].copy()
    df2["t"] = df2["timestamp"].astype(float)
    df2["logD"] = np.log(df2["difficulty"])

    X = df2[["t"]]
    y = df2["logD"]
    model = LinearRegression().fit(X, y)

    a = model.intercept_
    b = model.coef_[0]

    last = df2.iloc[-1]
    t0 = float(last["timestamp"])
    h0 = int(last["height"])
    D0 = float(last["difficulty"])

    return {
        "a": a,
        "b": b,
        "t0": t0,
        "h0": h0,
        "D0": D0,
        "df_fit": df2,
    }


def make_difficulty_func(D0, t0, b_scaled):
    def D(t):
        return D0 * np.exp(b_scaled * (t - t0))
    return D


def btc_price_powerlaw(dates, anchor_price_now):
    days = np.array([(dt - GENESIS).days for dt in dates], dtype=float)
    trend = PL_A * days**PL_B
    days_now = days[0]
    trend_now = PL_A * (days_now**PL_B)
    k = anchor_price_now / trend_now
    return k * trend


def simulate_miner(
    df_diff,
    difficulty_info,
    slope_factor,
    hashrate_ths,
    efficiency_j_per_th,
    equipment_price_usd,
    electricity_usd_per_kwh,
    btc_price_now_usd,
    years_horizon,
):
    # Difficulty model anchored at current point
    b_orig = difficulty_info["b"]
    t0 = difficulty_info["t0"]
    h0 = difficulty_info["h0"]
    D0 = difficulty_info["D0"]

    b_scaled = b_orig * slope_factor
    D_func = make_difficulty_func(D0, t0, b_scaled)

    # Epoch timing
    blocks_per_epoch = 2016
    blocks_per_day = 144
    days_per_epoch = blocks_per_epoch / blocks_per_day
    epoch_seconds = blocks_per_epoch * 600
    seconds_per_year = 365 * 24 * 3600

    total_seconds = years_horizon * seconds_per_year
    n_epochs = int(total_seconds // epoch_seconds) + 1

    epoch_idx = np.arange(n_epochs)
    timestamps = t0 + epoch_idx * epoch_seconds
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]
    heights = h0 + epoch_idx * blocks_per_epoch

    # Price curve
    btc_price = btc_price_powerlaw(dates, anchor_price_now=btc_price_now_usd)

    # Miner + power
    hashrate_hs = hashrate_ths * 1e12
    power_watts = hashrate_ths * efficiency_j_per_th
    power_kw = power_watts / 1000.0
    daily_kwh = power_kw * 24.0
    daily_elec_cost_usd = daily_kwh * electricity_usd_per_kwh
    elec_cost_epoch_usd = daily_elec_cost_usd * days_per_epoch

    sats_per_usd_now = SATS_PER_BTC / btc_price_now_usd
    equipment_cost_sats = equipment_price_usd * sats_per_usd_now

    difficulty = np.zeros(n_epochs)
    Hnet = np.zeros(n_epochs)
    share = np.zeros(n_epochs)
    subsidy_btc = np.zeros(n_epochs)
    btc_reward_epoch = np.zeros(n_epochs)
    sats_reward_epoch = np.zeros(n_epochs)
    net_sats_epoch = np.zeros(n_epochs)
    net_usd_epoch = np.zeros(n_epochs)

    for i in range(n_epochs):
        t = timestamps[i]
        difficulty[i] = D_func(t)
        Hnet[i] = difficulty[i] * (2**32) / 600.0
        share[i] = hashrate_hs / Hnet[i]

        h = heights[i]
        if h < 840_000:
            subsidy_btc[i] = 6.25
        elif h < 1_050_000:
            subsidy_btc[i] = 3.125
        elif h < 1_260_000:
            subsidy_btc[i] = 1.5625
        else:
            subsidy_btc[i] = 0.78125

        subsidy_sats_block = subsidy_btc[i] * SATS_PER_BTC
        reward_sats_block = subsidy_sats_block + FEE_SATS_PER_BLOCK
        reward_btc_block = reward_sats_block / SATS_PER_BTC

        btc_reward_epoch[i] = reward_btc_block * blocks_per_epoch * share[i]
        sats_reward_epoch[i] = btc_reward_epoch[i] * SATS_PER_BTC

        revenue_epoch_usd = btc_reward_epoch[i] * btc_price[i]
        net_usd = revenue_epoch_usd - elec_cost_epoch_usd
        net_usd_epoch[i] = net_usd

        if net_usd <= 0:
            net_sats_epoch[i] = 0.0
        else:
            sats_per_usd_here = SATS_PER_BTC / btc_price[i]
            elec_epoch_sats = elec_cost_epoch_usd * sats_per_usd_here
            net_sats_epoch[i] = sats_reward_epoch[i] - elec_epoch_sats

    cumulative_sats = -equipment_cost_sats + np.cumsum(net_sats_epoch)
    roi_indices = np.where(cumulative_sats >= 0)[0]
    roi_index = int(roi_indices[0]) if len(roi_indices) > 0 else None

    df_out = pd.DataFrame(
        {
            "epoch": epoch_idx,
            "timestamp": timestamps,
            "date": dates,
            "height": heights,
            "btc_price": btc_price,
            "difficulty": difficulty,
            "subsidy_btc": subsidy_btc,
            "sats_reward_epoch": sats_reward_epoch,
            "net_usd_epoch": net_usd_epoch,
            "net_sats_epoch": net_sats_epoch,
            "cumulative_sats": cumulative_sats,
        }
    )

    return df_out, equipment_cost_sats, roi_index


def main():
    if len(sys.argv) < 2:
        print("Usage: python mining_roi_sim.py rigs/rig_config.json")
        sys.exit(1)

    rig_config_path = sys.argv[1]
    rig = load_rig_config(rig_config_path)

    name = rig.get("name", "Unnamed rig")
    hashrate_ths = float(rig["hashrate_ths"])
    efficiency_j_per_th = float(rig["efficiency_j_per_th"])
    equipment_price_usd = float(rig["equipment_price_usd"])

    print(f"Rig: {name}")
    print(f"  Hashrate:   {hashrate_ths} TH/s")
    print(f"  Efficiency: {efficiency_j_per_th} J/TH")
    print(f"  Price:      ${equipment_price_usd}")

    # Load difficulty data
    df = pd.read_csv(CSV_PATH)

    # Fit difficulty
    diff_info = fit_difficulty_exp(df, DIFF_MIN_HEIGHT)
    print("Fitted difficulty slope b:", diff_info["b"])

    # Difficulty projections plot
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

    plt.figure(figsize=(12, 6))
    plt.plot(df_700k["height"], df_700k["difficulty"], label="Real difficulty (≥700k)")
    plt.plot(h_future, D_future_orig, "--", label="Projection: original slope")
    plt.plot(h_future, D_future_red, ":", label=f"Projection: reduced slope ({REDUCED_SLOPE_FACTOR:.2f}×)")
    plt.yscale("log")
    plt.xlabel("Block height")
    plt.ylabel("Difficulty (log scale)")
    plt.title("Bitcoin Difficulty: Real Data Since 700k vs Projections")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Mining sim – original slope
    df_orig, equip_sats, roi_orig = simulate_miner(
        df,
        diff_info,
        slope_factor=1.0,
        hashrate_ths=hashrate_ths,
        efficiency_j_per_th=efficiency_j_per_th,
        equipment_price_usd=equipment_price_usd,
        electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
        btc_price_now_usd=BTC_PRICE_NOW_USD,
        years_horizon=YEARS_HORIZON,
    )

    print("\nEquipment cost (sats):", equip_sats)
    print("Final cumulative sats (orig slope):", df_orig["cumulative_sats"].iloc[-1])
    print("ROI epoch index (orig slope):", roi_orig)

    # Mining sim – reduced slope
    df_red, _, roi_red = simulate_miner(
        df,
        diff_info,
        slope_factor=REDUCED_SLOPE_FACTOR,
        hashrate_ths=hashrate_ths,
        efficiency_j_per_th=efficiency_j_per_th,
        equipment_price_usd=equipment_price_usd,
        electricity_usd_per_kwh=ELECTRICITY_USD_PER_KWH,
        btc_price_now_usd=BTC_PRICE_NOW_USD,
        years_horizon=YEARS_HORIZON,
    )

    print("\nFinal cumulative sats (reduced slope):", df_red["cumulative_sats"].iloc[-1])
    print("ROI epoch index (reduced slope):", roi_red)

    # --- Combined ROI chart: original vs reduced difficulty slope ---
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_orig["date"],
        df_orig["cumulative_sats"],
        label="Cumulative sats – original slope",
    )
    plt.plot(
        df_red["date"],
        df_red["cumulative_sats"],
        linestyle="--",
        label=f"Cumulative sats – reduced slope ({REDUCED_SLOPE_FACTOR:.2f}×)",
    )
    plt.axhline(0, linestyle=":", label="Break-even (0 sats)")

    plt.xlabel("Date")
    plt.ylabel("Cumulative profit (sats)")
    plt.title(f"{name} – Cumulative Mining Profit (Two Difficulty Scenarios)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Price curve plot (matches ROI horizon) ---
    plt.figure(figsize=(12, 6))
    plt.plot(df_orig["date"], df_orig["btc_price"])
    plt.xlabel("Date")
    plt.ylabel("BTC price (USD)")
    plt.title(f"Projected BTC Price – Same Horizon as ROI Sim ({name})")
    plt.grid(True, which="both", axis="y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
