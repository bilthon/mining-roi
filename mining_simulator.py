from datetime import datetime, timezone
from typing import Optional, Tuple
from typing import Tuple

import numpy as np
import pandas as pd

from config import FEE_SATS_PER_BLOCK, HOURS_PER_WEEK
from difficulty_model import btc_price_powerlaw, make_difficulty_func

SATS_PER_BTC = 100_000_000

def simulate_miner(
    difficulty_info: dict,
    slope_factor: float,
    hashrate_ths: float,
    efficiency_j_per_th: float,
    equipment_price_usd: float,
    electricity_usd_per_kwh: float,
    btc_price_now_usd: float,
    years_horizon: int,
    curtailment_enabled: bool = False,
    curtailment_hours_per_week: float = HOURS_PER_WEEK,
    curtailment_electricity_usd_per_kwh: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, Optional[int], Optional[int]]:
    b_orig = difficulty_info["b"]
    t0 = difficulty_info["t0"]
    h0 = difficulty_info["h0"]
    D0 = difficulty_info["D0"]

    b_scaled = b_orig * slope_factor
    D_func = make_difficulty_func(D0, t0, b_scaled)

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

    btc_price = btc_price_powerlaw(dates, anchor_price_now=btc_price_now_usd)

    if curtailment_electricity_usd_per_kwh is None:
        curtailment_electricity_usd_per_kwh = electricity_usd_per_kwh

    if curtailment_enabled:
        uptime_hours = max(0.0, min(curtailment_hours_per_week, HOURS_PER_WEEK))
        effective_rate = curtailment_electricity_usd_per_kwh
    else:
        uptime_hours = HOURS_PER_WEEK
        effective_rate = electricity_usd_per_kwh

    uptime_fraction = uptime_hours / HOURS_PER_WEEK

    hashrate_hs = hashrate_ths * 1e12
    power_watts = hashrate_ths * efficiency_j_per_th
    power_kw = power_watts / 1000.0
    daily_kwh = power_kw * 24.0
    daily_elec_cost_usd = daily_kwh * effective_rate
    elec_cost_epoch_usd = daily_elec_cost_usd * days_per_epoch * uptime_fraction

    sats_per_usd_now = SATS_PER_BTC / btc_price_now_usd
    equipment_cost_sats = equipment_price_usd * sats_per_usd_now
    equipment_cost_usd = equipment_price_usd

    difficulty = np.zeros(n_epochs)
    Hnet = np.zeros(n_epochs)
    share = np.zeros(n_epochs)
    subsidy_btc = np.zeros(n_epochs)
    sats_reward_epoch = np.zeros(n_epochs)
    net_sats_epoch = np.zeros(n_epochs)
    net_usd_epoch = np.zeros(n_epochs)
    daily_net_sats = np.zeros(n_epochs)
    daily_net_usd = np.zeros(n_epochs)
    btc_reward_epoch = np.zeros(n_epochs)

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

        btc_reward_epoch[i] = (
            reward_btc_block * blocks_per_epoch * share[i] * uptime_fraction
        )
        sats_reward_epoch[i] = btc_reward_epoch[i] * SATS_PER_BTC

        revenue_epoch_usd = btc_reward_epoch[i] * btc_price[i]
        net_usd = revenue_epoch_usd - elec_cost_epoch_usd
        net_usd_epoch[i] = net_usd

        if net_usd <= 0:
            net_usd_epoch[i] = 0.0
            net_sats_epoch[i] = 0.0
            daily_net_usd[i] = 0.0
            daily_net_sats[i] = 0.0
        else:
            sats_per_usd_here = SATS_PER_BTC / btc_price[i]
            elec_epoch_sats = elec_cost_epoch_usd * sats_per_usd_here
            net_sats_epoch[i] = sats_reward_epoch[i] - elec_epoch_sats
            daily_net_usd[i] = net_usd / days_per_epoch
            daily_net_sats[i] = net_sats_epoch[i] / days_per_epoch

    cumulative_sats = -equipment_cost_sats + np.cumsum(net_sats_epoch)
    roi_indices = np.where(cumulative_sats >= 0)[0]
    roi_sats_index = int(roi_indices[0]) if len(roi_indices) > 0 else None

    cumulative_usd = -equipment_cost_usd + np.cumsum(net_usd_epoch)
    roi_usd_idx = np.where(cumulative_usd >= 0)[0]
    roi_usd_idx = int(roi_usd_idx[0]) if len(roi_usd_idx) > 0 else None

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
            "daily_net_usd": daily_net_usd,
            "daily_net_sats": daily_net_sats,
            "cumulative_sats": cumulative_sats,
            "cumulative_usd": cumulative_usd,
        }
    )

    return df_out, equipment_cost_sats, roi_sats_index, roi_usd_idx