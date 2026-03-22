set -e


echo "Running with $@"
python scripts/prepare_data.py --config-name=$config "$@"
python scripts/benchmark_single.py --config-name=$config "$@"
python scripts/benchmark_e2e.py --config-name=$config "$@"
python scripts/plot_results.py --config-name=$config "$@"
