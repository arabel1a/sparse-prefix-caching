overrides=$@

set -e

echo "Runnig with overrides $overrides"
python scripts/prepare_data.py $overrides
python scripts/benchmark_single.py $overrides
python scripts/benchmark_e2e.py $overrides
python scripts/plot_results.py $overrides
