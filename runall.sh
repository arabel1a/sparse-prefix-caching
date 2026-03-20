overrides=$@

set -e

uv sync

echo "Runnig with overrides $overrides"
uv run python scripts/prepare_data.py $overrides
uv run python scripts/benchmark_single.py $overrides
uv run python scripts/benchmark_e2e.py $overrides
uv run python scripts/plot_results.py $overrides
