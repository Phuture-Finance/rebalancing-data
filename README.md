# Rebalancing Data

This repository collects methodology notebooks and utilities for designing and backtesting
cryptocurrency indices.  The project centres around **Phuture** index methodologies and
stores historical rebalancing calculations.

## Contents

- **Index folders** (`AEX/`, `BEX/`, `CAI/`, etc.) – each folder contains a
  `methodology_template.ipynb` or dated notebooks that detail the rules and the
  monthly rebalancing for a specific index.
- **methodology.py** – core library providing `MethodologyBase` and `MethodologyProd`
  classes for gathering market data, applying eligibility rules and computing
  asset weights.
- **backtester.py** – script for evaluating index performance and rebalancing
  schedules.
- **db_funcs.py** – SQLite helper functions used to persist benchmarking and
  liquidity data.
- **abis.py** – contract ABIs used for on‑chain interactions.
- `Pipfile` / `Pipfile.lock` – Python dependencies (requires Python 3.13+).

Data is stored in (`rebalancing_data_db.sqlite`) 

## Installation

Use `pipenv` or `pip` to install the required packages.  With pipenv:

```bash
pipenv install --dev
```

or with pip:

```bash
pip install -r <(pipenv lock -r)
```

## Environment

Several API keys and endpoints are read from environment variables:

- `CG_KEY` – CoinGecko API key.
- `ZEROEX_KEY` – 0x API key used for slippage estimates.
- Blockchain RPC URLs such as `ETHEREUM_INFURA_URL`, `AVALANCHE_INFURA_URL`,
  `POLYGON_INFURA_URL`, etc.

Set these variables before running the notebooks or scripts.

## Usage

The notebooks demonstrate how to instantiate a `MethodologyProd` object,
fetch market data and generate the final weights.  The `backtester.py` module
can simulate rebalancing strategies and compare them against benchmarks:

```python
from backtester import BackTester

# Example asset weights
weights = {"bitcoin": 0.5, "ethereum": 0.5}
bench = ["bitcoin"]

bt = BackTester(weights, bench)
fig, values, returns = bt.simulate_with_rebalancing(30, liquidity_discount=0)
fig.show()
```

The outputs include performance charts and cumulative returns for the chosen
rebalancing window.

## License

This project is distributed under the terms of the GNU General Public
License v3.0.  See the [LICENSE](LICENSE) file for details.
