from datetime import datetime, timezone
from pathlib import Path

# Try to load TOML config, fall back to defaults if not found
# BASE_DIR should point at the project root, not the src/ folder
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_TOML = BASE_DIR / "config.toml"

try:
    # Try tomllib (Python 3.11+)
    try:
        import tomllib # type: ignore
        with open(CONFIG_TOML, "rb") as f:
            config = tomllib.load(f)
    except ImportError:
        # Fall back to tomli or toml for older Python versions
        try:
            import tomli as tomllib # type: ignore
            with open(CONFIG_TOML, "rb") as f:
                config = tomllib.load(f)
        except ImportError:
            import toml # type: ignore
            with open(CONFIG_TOML, "r") as f:
                config = toml.load(f)
    
    # Load configuration from TOML
    CSV_PATH = BASE_DIR / config["paths"]["csv_path"]
    RIGS_DIR = BASE_DIR / config["paths"]["rigs_dir"]
    
    ELECTRICITY_USD_PER_KWH = float(config["economics"]["electricity_usd_per_kwh"])
    BTC_PRICE_NOW_USD = float(config["economics"]["btc_price_now_usd"])
    FEE_SATS_PER_BLOCK = int(config["economics"]["fee_sats_per_block"])
    
    YEARS_HORIZON = int(config["simulation"]["years_horizon"])
    DIFF_MIN_HEIGHT = int(config["simulation"]["diff_min_height"])
    REDUCED_SLOPE_FACTOR = float(config["simulation"]["reduced_slope_factor"])
    
    HOURS_PER_WEEK = 7 * 24
    CURTAILMENT_ENABLED = bool(config["curtailment"]["enabled"])
    CURTAILMENT_HOURS_PER_WEEK = int(config["curtailment"]["hours_per_week"])
    CURTAILMENT_ELECTRICITY_USD_PER_KWH = float(config["curtailment"]["electricity_usd_per_kwh"])
    
    PL_A = float(config["power_law"]["a"])
    PL_B = float(config["power_law"]["b"])
    
    # Parse genesis date
    genesis_str = config["genesis"]["date"]
    if genesis_str.endswith("Z"):
        genesis_str = genesis_str[:-1] + "+00:00"
    GENESIS = datetime.fromisoformat(genesis_str)
    
except (FileNotFoundError, ImportError, KeyError):
    # Default configuration values (fallback)
    CSV_PATH = BASE_DIR / "data" / "difficulty_epochs.csv"
    RIGS_DIR = BASE_DIR / "rigs"

    ELECTRICITY_USD_PER_KWH = 0.05
    BTC_PRICE_NOW_USD = 90_000.0

    FEE_SATS_PER_BLOCK = 2_000_000
    YEARS_HORIZON = 4
    DIFF_MIN_HEIGHT = 700_000

    REDUCED_SLOPE_FACTOR = 0.75

    HOURS_PER_WEEK = 7 * 24
    CURTAILMENT_ENABLED = False
    CURTAILMENT_HOURS_PER_WEEK = 144
    CURTAILMENT_ELECTRICITY_USD_PER_KWH = 0.041

    PL_A = 1.44e-17
    PL_B = 5.78
    GENESIS = datetime(2009, 1, 3, tzinfo=timezone.utc)
