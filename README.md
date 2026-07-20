# MeldBenchmark

Insert manifesto here.

## Setup

We use `mamba` and `uv` for lightning-fast dependency management.

1. Create the environment:
   ```bash
   mamba env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   mamba activate meld
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

Run the data inspection script to examine local Zarr bundles and remote CSV tracking data:
```bash
python src/inspect_tracking_data.py
```

## Development

This project uses `ruff` for linting and formatting. It's recommended to use `pre-commit` to ensure code quality:
```bash
pre-commit install
```
