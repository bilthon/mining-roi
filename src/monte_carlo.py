from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    BTC_PRICE_NOW_USD,
    CURTAILMENT_ELECTRICITY_USD_PER_KWH,
    CURTAILMENT_ENABLED,
    CURTAILMENT_HOURS_PER_WEEK,
    ELECTRICITY_USD_PER_KWH,
    YEARS_HORIZON,
)
from .data_loader import load_rig_config
from .mining_simulator import simulate_miner
from .residual_analysis import sample_residuals


def _build_timeline(diff_info: Dict, years_horizon: int) -> Dict[str, np.ndarray]:
    blocks_per_epoch = 2016
    blocks_per_day = 144
    days_per_epoch = blocks_per_epoch / blocks_per_day
    epoch_seconds = blocks_per_epoch * 600
    seconds_per_year = 365 * 24 * 3600

    t0 = diff_info["t0"]
    h0 = diff_info["h0"]

    total_seconds = years_horizon * seconds_per_year
    n_epochs = int(total_seconds // epoch_seconds) + 1

    epoch_idx = np.arange(n_epochs)
    timestamps = t0 + epoch_idx * epoch_seconds
    heights = h0 + epoch_idx * blocks_per_epoch

    return {
        "epoch_idx": epoch_idx,
        "timestamps": timestamps,
        "heights": heights,
        "days_per_epoch": days_per_epoch,
    }


def generate_difficulty_paths(
    diff_info: Dict,
    years_horizon: int = YEARS_HORIZON,
    n_sims: int = 1,
    residuals: Optional[Sequence[float]] = None,
    slope_factor: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if residuals is None:
        residuals = diff_info.get("residuals")
    if residuals is None:
        raise ValueError("Residuals are required to generate stochastic difficulty paths.")

    timeline = _build_timeline(diff_info, years_horizon)
    timestamps = timeline["timestamps"]
    base_logD = np.log(diff_info["D0"]) + (diff_info["b"] * slope_factor) * (timestamps - diff_info["t0"])

    rng = np.random.default_rng(seed)
    difficulty_paths = np.zeros((n_sims, base_logD.shape[0]))
    for i in range(n_sims):
        sim_seed = int(rng.integers(0, 1_000_000_000))
        res = sample_residuals(
            n=base_logD.shape[0],
            residuals=residuals,
            method="bootstrap",
            seed=sim_seed,
        )
        logD_path = base_logD + np.cumsum(res)
        difficulty_paths[i] = np.exp(logD_path)

    return difficulty_paths, timeline


def run_monte_carlo_for_rig(
    diff_info: Dict,
    rig: Dict,
    years_horizon: int = YEARS_HORIZON,
    n_sims: int = 100,
    slope_factor: float = 1.0,
    seed: Optional[int] = None,
    btc_price_now_usd: float = BTC_PRICE_NOW_USD,
    electricity_usd_per_kwh: float = ELECTRICITY_USD_PER_KWH,
    curtailment_enabled: bool = CURTAILMENT_ENABLED,
    curtailment_hours_per_week: float = CURTAILMENT_HOURS_PER_WEEK,
    curtailment_electricity_usd_per_kwh: Optional[float] = CURTAILMENT_ELECTRICITY_USD_PER_KWH,
) -> Dict:
    difficulty_paths, timeline = generate_difficulty_paths(
        diff_info=diff_info,
        years_horizon=years_horizon,
        n_sims=n_sims,
        slope_factor=slope_factor,
        seed=seed,
    )

    hashrate_ths = float(rig["hashrate_ths"])
    efficiency_j_per_th = float(rig["efficiency_j_per_th"])
    equipment_price_usd = float(rig["equipment_price_usd"])

    sim_results: List[Dict] = []
    final_sats = []
    final_usd = []
    roi_sats_epochs = []
    roi_usd_epochs = []

    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timeline["timestamps"]]

    for idx, difficulty_path in enumerate(difficulty_paths):
        df_out, equip_sats, roi_sats, roi_usd = simulate_miner(
            difficulty_info=diff_info,
            slope_factor=slope_factor,
            hashrate_ths=hashrate_ths,
            efficiency_j_per_th=efficiency_j_per_th,
            equipment_price_usd=equipment_price_usd,
            electricity_usd_per_kwh=electricity_usd_per_kwh,
            btc_price_now_usd=btc_price_now_usd,
            years_horizon=years_horizon,
            curtailment_enabled=curtailment_enabled,
            curtailment_hours_per_week=curtailment_hours_per_week,
            curtailment_electricity_usd_per_kwh=curtailment_electricity_usd_per_kwh,
            difficulty_override=difficulty_path,
            timestamps_override=timeline["timestamps"],
            heights_override=timeline["heights"],
            dates_override=dates,
        )

        final_sats.append(df_out["cumulative_sats"].iloc[-1])
        final_usd.append(df_out["cumulative_usd"].iloc[-1])
        roi_sats_epochs.append(roi_sats)
        roi_usd_epochs.append(roi_usd)

        sim_results.append(
            {
                "simulation": idx,
                "df": df_out,
                "equipment_cost_sats": equip_sats,
                "roi_sats_epoch": roi_sats,
                "roi_usd_epoch": roi_usd,
                "final_cumulative_sats": df_out["cumulative_sats"].iloc[-1],
                "final_cumulative_usd": df_out["cumulative_usd"].iloc[-1],
            }
        )

    def _percentiles(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    summary = {
        "final_sats": _percentiles(final_sats),
        "final_usd": _percentiles(final_usd),
        "roi_sats_epochs": [e for e in roi_sats_epochs if e is not None],
        "roi_usd_epochs": [e for e in roi_usd_epochs if e is not None],
    }

    return {
        "timeline": timeline,
        "difficulty_paths": difficulty_paths,
        "sim_results": sim_results,
        "summary": summary,
    }


def load_rig_and_run_monte_carlo(
    diff_info: Dict,
    rig_path: Path,
    years_horizon: int = YEARS_HORIZON,
    n_sims: int = 100,
    slope_factor: float = 1.0,
    seed: Optional[int] = None,
    btc_price_now_usd: float = BTC_PRICE_NOW_USD,
    electricity_usd_per_kwh: float = ELECTRICITY_USD_PER_KWH,
    curtailment_enabled: bool = CURTAILMENT_ENABLED,
    curtailment_hours_per_week: float = CURTAILMENT_HOURS_PER_WEEK,
    curtailment_electricity_usd_per_kwh: Optional[float] = CURTAILMENT_ELECTRICITY_USD_PER_KWH,
) -> Dict:
    rig = load_rig_config(rig_path)
    return run_monte_carlo_for_rig(
        diff_info=diff_info,
        rig=rig,
        years_horizon=years_horizon,
        n_sims=n_sims,
        slope_factor=slope_factor,
        seed=seed,
        btc_price_now_usd=btc_price_now_usd,
        electricity_usd_per_kwh=electricity_usd_per_kwh,
        curtailment_enabled=curtailment_enabled,
        curtailment_hours_per_week=curtailment_hours_per_week,
        curtailment_electricity_usd_per_kwh=curtailment_electricity_usd_per_kwh,
    )
