set -e

config=${1:-config}
shift 2>/dev/null || true

echo "Running with config=$config $@"
uv run python scripts/prepare_data.py --config-name=$config "$@"
uv run python scripts/benchmark_single.py --config-name=$config "$@"
uv run python scripts/benchmark_e2e.py --config-name=$config "$@"
uv run python scripts/plot_results.py --config-name=$config "$@"
