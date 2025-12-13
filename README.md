# Bitcoin Mining ROI Simulator

A comprehensive Python tool for simulating Bitcoin mining profitability and return on investment (ROI) over time. This simulator models Bitcoin difficulty growth, projects BTC prices, and calculates mining profitability for various mining rigs, helping you make informed decisions about mining equipment investments.

## Features

- **Monte Carlo-first difficulty**: Stochastic difficulty paths via log-space random walk on fitted residuals (primary analysis)
- **Difficulty modeling**: Exponential regression fit to historical data as the drift for Monte Carlo paths
- **Price projections**: Power law model anchored to current prices
- **Multi-rig Monte Carlo comparison**: Run the same difficulty scenarios across multiple rigs for fair comparisons
- **ROI distributions**: Percentiles for final sats/USD and ROI-epoch distributions
- **Visualizations**: ROI clouds, sample paths, difficulty-path overlays, and price forecasts
- **Flexible configuration**: Easily add new mining rigs via JSON configuration files

## Project Structure

```
btc-difficulty-mining/
├── config.toml.sample           # Sample configuration file (copy to config.toml)
├── config.toml                  # Local configuration (gitignored, create from sample)
├── main.py                      # Main CLI script for running simulations
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration loader (loads from config.toml)
│   ├── data_loader.py           # Functions to load rig configs and difficulty data
│   ├── difficulty_model.py      # Difficulty fitting and BTC price projection models
│   ├── mining_simulator.py      # Core mining profitability simulation logic
│   └── plotting.py              # Visualization functions
├── scripts/
│   └── export_difficulty_epochs.py  # Script to export difficulty data from Bitcoin Core
├── data/
│   └── difficulty_epochs.csv    # Historical Bitcoin difficulty data
└── rigs/                        # Mining rig configuration files (JSON)
    ├── s19_xp.json
    ├── s19j_pro.json
    ├── s21_plus_225th.json
    └── ...
```

## Installation

### Prerequisites

- Python 3.7+ (Python 3.11+ recommended for built-in TOML support)
- Bitcoin Core (optional, only needed for exporting difficulty data)

### Dependencies

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn requests
```

For Python versions < 3.11, you'll also need a TOML library:

```bash
pip install tomli  # For Python 3.7-3.10
# OR
pip install toml   # Alternative TOML library
```

Or create a `requirements.txt` file:

```txt
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
requests>=2.25.0
tomli>=2.0.0; python_version < "3.11"
```

Then install:

```bash
pip install -r requirements.txt
```

## Configuration

### Initial Setup

On first deployment, copy the sample configuration file:

```bash
cp config.toml.sample config.toml
```

Then edit `config.toml` with your preferred settings. The `config.toml` file is gitignored, so your local changes won't be tracked by git.

### Main Configuration (`config.toml`)

Key parameters you can adjust in `config.toml`:

The configuration file uses TOML format with the following sections:

**`[paths]`**
- `csv_path`: Path to difficulty epochs CSV file (relative to project root, default: `data/difficulty_epochs.csv`)
- `rigs_dir`: Directory containing rig configuration JSON files

**`[economics]`**
- `electricity_usd_per_kwh`: Your electricity cost (default: $0.05/kWh)
- `btc_price_now_usd`: Current BTC price for anchoring projections (default: $90,000)
- `fee_sats_per_block`: Average transaction fees per block in satoshis (default: 2,000,000)

**`[simulation]`**
- `years_horizon`: Simulation time horizon (default: 4 years)
- `diff_min_height`: Minimum block height for difficulty fitting (default: 700,000)
- `default_mc_simulations`: Default number of Monte Carlo simulations (default: 100)
- `mc_default_seed`: Optional default RNG seed for Monte Carlo (can be omitted)
- `mc_default_bands`: Default percentile bands for ROI plots (e.g., `"10-90,25-75"`)

**`[curtailment]`**
- `enabled`: Enable curtailed uptime modeling (default: `false`)
- `hours_per_week`: Number of hours each week the miner can operate when curtailment is enabled (default: 144)
- `electricity_usd_per_kwh`: Electricity price applied during the allowed runtime window (default: $0.041/kWh)

**`[power_law]`**
- `a`: Power law constant A for BTC price projection (default: 1.44e-17)
- `b`: Power law constant B for BTC price projection (default: 5.78)

**`[genesis]`**
- `date`: Bitcoin genesis block date in ISO format (default: "2009-01-03T00:00:00Z")

To represent a 120-hour cheap-power window, set `[curtailment] enabled = true`, `hours_per_week = 120`, and assign the discounted rate to `electricity_usd_per_kwh`.

### Mining Rig Configuration

Rig configurations are stored as JSON files in the `rigs/` directory. Each file should contain:

```json
{
  "name": "Rig Name",
  "hashrate_ths": 225.0,
  "efficiency_j_per_th": 16.5,
  "equipment_price_usd": 4500.0
}
```

**Fields:**
- `name`: Human-readable name for the rig
- `hashrate_ths`: Hashrate in terahashes per second (TH/s)
- `efficiency_j_per_th`: Energy efficiency in joules per terahash (J/TH)
- `equipment_price_usd`: Equipment purchase price in USD

## Usage

### Basic Usage

Run a simulation for all rigs in the `rigs/` directory:

```bash
python main.py
```

Run a simulation for a specific rig:

```bash
python main.py rigs/s21_plus_225th.json
```

### Command-Line Options

```bash
python main.py [rig_config] [options]
```

**Arguments:**
- `rig_config` (optional): Path to a specific rig configuration JSON file

**Options:**
- `--rigs-dir DIR`: Specify a custom directory containing rig JSON configs (default: `rigs/`)
- `--diff`: Overlay Monte Carlo difficulty paths on historical difficulty
- `--price`: Include BTC price projection plot
- `--n-sims N`: Number of Monte Carlo simulations (default from config)
- `--mc-seed SEED`: Optional RNG seed for Monte Carlo sampling
- `--mc-show-paths N`: Overlay N individual Monte Carlo cumulative-sats paths on the ROI plot
- `--mc-bands LIST`: Comma-separated percentile bands for ROI cloud (e.g., `10-90,25-75`)
- `--mc-show-difficulty N`: Overlay N Monte Carlo difficulty trajectories on the historical difficulty step plot

### Examples

Single rig with default simulations:

```bash
python main.py rigs/s21_plus_225th.json
```

Single rig with custom simulation count:

```bash
python main.py rigs/s21_plus_225th.json --n-sims 200 --mc-seed 42
```

Multi-rig comparison (uses shared difficulty paths):

```bash
python main.py --n-sims 50
```

Show sample ROI paths and custom bands:

```bash
python main.py rigs/s21_plus_225th.json --mc-show-paths 5 --mc-bands "5-95,25-75"
```

Show Monte Carlo difficulty trajectories on top of history:

```bash
python main.py rigs/s21_plus_225th.json --mc-show-difficulty 5 --diff
```

Use a custom rigs directory:

```bash
python main.py --rigs-dir ~/my-rigs
```

## Exporting Difficulty Data

If you have Bitcoin Core running with RPC enabled, you can export difficulty data:

```bash
python scripts/export_difficulty_epochs.py [output.csv]
```

**Environment Variables:**
- `BTC_CHAIN`: Bitcoin chain (mainnet, testnet, regtest) - default: `mainnet`
- `BTC_RPC_HOST`: RPC host - default: `127.0.0.1`
- `BTC_RPC_PORT`: RPC port - default: `8332`

The script reads authentication from `~/.bitcoin/.cookie` (or testnet/regtest equivalents).

## How It Works

### Difficulty Modeling

The simulator fits an exponential model to historical difficulty data:

```
D(t) = D₀ × exp(b × (t - t₀))
```

Where:
- `D(t)` is difficulty at time `t`
- `D₀` is the difficulty at the reference time `t₀`
- `b` is the growth rate (fitted from historical data)

The model uses data from block height 700,000 onwards (configurable via `DIFF_MIN_HEIGHT`).

### Price Projection

BTC prices are projected using a power law model:

```
P(days) = k × A × days^B
```

Where `A` and `B` are power law constants, and `k` is a scaling factor to anchor the projection to the current BTC price.

### Mining Simulation

For each epoch (2016 blocks ≈ 2 weeks):

1. **Calculate network hashrate**: `H_net = D × 2³² / 600`
2. **Calculate miner's share**: `share = hashrate_miner / H_net`
3. **Calculate rewards**: 
   - Block subsidy (halving-aware: 6.25 → 3.125 → 1.5625 → 0.78125 BTC)
   - Transaction fees (configurable)
4. **Calculate costs**: Electricity consumption based on hashrate and efficiency
5. **Apply curtailment (optional)**: If enabled, rewards and electricity costs are scaled by the uptime fraction (e.g., 120/168 hours per week) using the specified curtailed electricity rate.
6. **Calculate net profit**: Revenue - electricity costs
7. **Track cumulative profit**: Starting from negative equipment cost

The simulation accounts for:
- Difficulty increases over time
- BTC price changes
- Halving events (subsidy reductions)
- Electricity costs
- Equipment depreciation (initial cost)

### ROI Calculation

ROI is calculated in two ways:
- **Satoshis**: When cumulative satoshis exceed equipment cost (in sats)
- **USD**: When cumulative USD profit exceeds equipment cost

The simulator shows when break-even is reached for both metrics.

## Output

The simulator provides:

1. **Console Output**:
   - Rig specifications (hashrate, efficiency, price)
   - Equipment cost in satoshis
   - Final cumulative profits percentiles (sats and USD)
   - ROI epoch percentiles (when break-even is reached)

2. **Visualizations**:
   - **Single Rig Analysis**:
     - Monte Carlo ROI cloud with percentile bands (cumulative sats)
     - Sample cumulative-sats paths (optional)
     - Monte Carlo difficulty path overlays on historical difficulty (optional)
     - BTC price projection (optional)
   
   - **Multi-Rig Comparison**:
     - Percentile bands and median lines per rig (shared difficulty scenarios)
     - Monte Carlo difficulty overlays (optional)
     - BTC price projection (optional, sample path)

## Understanding the Results

### Monte Carlo (random-walk) Difficulty

- Uses the fitted exponential slope as drift and adds a cumulative random walk in log-difficulty using bootstrapped residuals from the historical fit (starts at `log(D0)`).
- Each simulation yields a full difficulty path that feeds the ROI simulator; BTC price modeling and halving logic are unchanged.
- Outputs include percentile bands of cumulative sats over time and a distribution of ROI epochs (when cumulative sats crosses zero) across simulations.
- In multi-rig mode, all rigs share the same difficulty paths to keep comparisons fair.

### Key Metrics

- **ROI Epoch Index**: The epoch number when cumulative profit reaches zero (break-even point)
- **Final Cumulative Profit**: Total profit at the end of the simulation horizon
- **Daily Net Profit**: Daily profit after electricity costs (can become negative if electricity costs exceed revenue)

### Important Considerations

- **Assumptions**: The model assumes constant electricity costs, no equipment failures, and continuous mining
- **Curtailment Windows**: With curtailment enabled, the miner only earns revenue and incurs electricity costs during the configured uptime window (e.g., 120 cheap-rate hours each week); outside that window it is fully powered down.
- **Price Volatility**: BTC price projections are based on historical trends and may not reflect future volatility
- **Difficulty Growth**: Actual difficulty growth may differ from projections
- **Network Fees**: Uses a fixed average fee per block; actual fees vary
- **No Pool Fees**: Pool fees are not included in calculations

## Adding New Rigs

To add a new mining rig:

1. Create a JSON file in the `rigs/` directory
2. Use the following template:

```json
{
  "name": "Your Rig Name",
  "hashrate_ths": 0.0,
  "efficiency_j_per_th": 0.0,
  "equipment_price_usd": 0.0
}
```

3. Fill in the values:
   - `hashrate_ths`: Your rig's hashrate in TH/s
   - `efficiency_j_per_th`: Energy efficiency in J/TH (lower is better)
   - `equipment_price_usd`: Purchase price in USD

4. The rig will automatically be included when running multi-rig comparisons

## Contributing

Contributions are welcome! Areas for improvement:

- More sophisticated difficulty models
- Variable electricity pricing
- Risk analysis and Monte Carlo simulations

## License

This project is provided as-is for educational and research purposes. Use at your own risk when making investment decisions.

