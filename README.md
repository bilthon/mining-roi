# Bitcoin Mining ROI Simulator

A comprehensive Python tool for simulating Bitcoin mining profitability and return on investment (ROI) over time. This simulator models Bitcoin difficulty growth, projects BTC prices, and calculates mining profitability for various mining rigs, helping you make informed decisions about mining equipment investments.

## Features

- **Difficulty Modeling**: Uses exponential regression to model Bitcoin difficulty growth based on historical data
- **Price Projections**: Projects BTC prices using a power law model anchored to current prices
- **Multi-Rig Comparison**: Compare profitability across different mining rigs simultaneously
- **Scenario Analysis**: Evaluate two difficulty growth scenarios (original slope vs. reduced slope)
- **Comprehensive Metrics**: Track ROI in both satoshis and USD, including daily profit projections
- **Visualizations**: Generate detailed charts for difficulty projections, cumulative profits, daily profits, and price forecasts
- **Flexible Configuration**: Easily add new mining rigs via JSON configuration files

## Project Structure

```
btc-difficulty-mining/
├── config.py                    # Configuration constants (electricity costs, BTC price, etc.)
├── data_loader.py               # Functions to load rig configs and difficulty data
├── difficulty_model.py          # Difficulty fitting and BTC price projection models
├── mining_simulator.py          # Core mining profitability simulation logic
├── mining_roi_sim.py            # Main CLI script for running simulations
├── plotting.py                  # Visualization functions
├── export_difficulty_epochs.py  # Script to export difficulty data from Bitcoin Core
├── difficulty_epochs.csv        # Historical Bitcoin difficulty data
└── rigs/                        # Mining rig configuration files (JSON)
    ├── s19_xp.json
    ├── s19j_pro.json
    ├── s21_plus_225th.json
    └── ...
```

## Installation

### Prerequisites

- Python 3.7+
- Bitcoin Core (optional, only needed for exporting difficulty data)

### Dependencies

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn requests
```

Or create a `requirements.txt` file:

```txt
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
requests>=2.25.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Configuration

### Main Configuration (`config.py`)

Key parameters you can adjust in `config.py`:

- `ELECTRICITY_USD_PER_KWH`: Your electricity cost (default: $0.05/kWh)
- `BTC_PRICE_NOW_USD`: Current BTC price for anchoring projections (default: $90,000)
- `YEARS_HORIZON`: Simulation time horizon (default: 4 years)
- `DIFF_MIN_HEIGHT`: Minimum block height for difficulty fitting (default: 700,000)
- `REDUCED_SLOPE_FACTOR`: Factor for reduced difficulty growth scenario (default: 0.75)
- `FEE_SATS_PER_BLOCK`: Average transaction fees per block in satoshis (default: 1,000,000)

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
python mining_roi_sim.py
```

Run a simulation for a specific rig:

```bash
python mining_roi_sim.py rigs/s21_plus_225th.json
```

### Command-Line Options

```bash
python mining_roi_sim.py [rig_config] [options]
```

**Arguments:**
- `rig_config` (optional): Path to a specific rig configuration JSON file

**Options:**
- `--rigs-dir DIR`: Specify a custom directory containing rig JSON configs (default: `rigs/`)
- `--diff`: Include difficulty projection plots
- `--price`: Include BTC price projection plots

### Examples

Compare all rigs with difficulty and price projections:

```bash
python mining_roi_sim.py --diff --price
```

Analyze a specific rig with all visualizations:

```bash
python mining_roi_sim.py rigs/s21_plus_225th.json --diff --price
```

Use a custom rigs directory:

```bash
python mining_roi_sim.py --rigs-dir ~/my-rigs
```

## Exporting Difficulty Data

If you have Bitcoin Core running with RPC enabled, you can export difficulty data:

```bash
python export_difficulty_epochs.py [output.csv]
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
5. **Calculate net profit**: Revenue - electricity costs
6. **Track cumulative profit**: Starting from negative equipment cost

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
   - Final cumulative profits (sats and USD)
   - ROI epoch indices (when break-even is reached)

2. **Visualizations**:
   - **Single Rig Analysis**:
     - Cumulative profit in satoshis (two difficulty scenarios)
     - Cumulative profit in USD (two difficulty scenarios)
     - Daily profit (sats and USD)
     - BTC price projection (optional)
   
   - **Multi-Rig Comparison**:
     - Cumulative sats comparison
     - Cumulative USD comparison
     - BTC price projection (optional)
   
   - **Difficulty Projections** (with `--diff`):
     - Historical difficulty vs. projections (log and linear scales)
     - Original slope vs. reduced slope scenarios

## Understanding the Results

### Difficulty Scenarios

The simulator runs two scenarios:

1. **Original Slope**: Uses the fitted difficulty growth rate directly
2. **Reduced Slope**: Uses a reduced growth rate (default: 75% of original)

The reduced slope scenario provides a more conservative estimate, accounting for potential slowdowns in difficulty growth.

### Key Metrics

- **ROI Epoch Index**: The epoch number when cumulative profit reaches zero (break-even point)
- **Final Cumulative Profit**: Total profit at the end of the simulation horizon
- **Daily Net Profit**: Daily profit after electricity costs (can become negative if electricity costs exceed revenue)

### Important Considerations

- **Assumptions**: The model assumes constant electricity costs, no equipment failures, and continuous mining
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

