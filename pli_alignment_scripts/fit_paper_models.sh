#!/usr/bin/env bash
set -euo pipefail

# Default: execute commands
DRY_RUN=false

# Parse options
while getopts "v" opt; do
  case $opt in
    v) DRY_RUN=true ;;
    *) echo "Usage: $0 [-v]" >&2; exit 1 ;;
  esac
done

# Helper to run or print commands
run_cmd() {
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    "$@"
  fi
}

# Arrays (fixed syntax)
split_types=("pocket" "scaffold")
folds=(0 1 2 3 4)

CGNNX_ARGS=(--hidden_channels 256 --num_heads 4 --num_attention_blocks 3)
DIMENET_ARGS=(--hidden_channels 256 --lr 0.0001)

for split_type in "${split_types[@]}"; do
  for fold in "${folds[@]}"; do
    cgnn_name="cgnn_${split_type}_${fold}"
    cgnn3d_name="cgnn3d_${split_type}_${fold}"
    dimenet_name="dimenet_${split_type}_${fold}"

    run_cmd ./fit_model.sh train_sparse_transformer "$cgnn_name" \
      --split_type "${split_type}-k-fold" \
      --split_index "$fold" \
      --covalent_only True \
      "${CGNNX_ARGS[@]}"

    run_cmd ./fit_model.sh train_sparse_transformer "$cgnn3d_name" \
      --split_type "${split_type}-k-fold" \
      --split_index "$fold" \
      --covalent_only False \
      "${CGNNX_ARGS[@]}"

    run_cmd ./fit_model.sh train_dimenet "$dimenet_name" \
      --split_type "${split_type}-k-fold" \
      --split_index "$fold" \
      "${DIMENET_ARGS[@]}"
  done
done