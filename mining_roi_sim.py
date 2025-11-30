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
    heights

