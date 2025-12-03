from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import GENESIS, PL_A, PL_B


def fit_difficulty_exp(df: pd.DataFrame, min_height: int) -> dict:
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


def make_difficulty_func(D0: float, t0: float, b_scaled: float) -> Callable[[float], float]:
    def D(t: float) -> float:
        return D0 * np.exp(b_scaled * (t - t0))

    return D


def btc_price_powerlaw(dates: Iterable[datetime], anchor_price_now: float) -> np.ndarray:
    days = np.array([(dt - GENESIS).days for dt in dates], dtype=float)
    trend = PL_A * days**PL_B
    days_now = days[0]
    trend_now = PL_A * (days_now**PL_B)
    k = anchor_price_now / trend_now
    return k * trend

