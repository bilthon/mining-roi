from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).parent

CSV_PATH = BASE_DIR / "difficulty_epochs.csv"
RIGS_DIR = BASE_DIR / "rigs"

ELECTRICITY_USD_PER_KWH = 0.05
BTC_PRICE_NOW_USD = 90_000.0

FEE_SATS_PER_BLOCK = 1_000_000
YEARS_HORIZON = 4
DIFF_MIN_HEIGHT = 700_000

REDUCED_SLOPE_FACTOR = 0.75

PL_A = 1.44e-17
PL_B = 5.78
GENESIS = datetime(2009, 1, 3, tzinfo=timezone.utc)

SATS_PER_BTC = 100_000_000

