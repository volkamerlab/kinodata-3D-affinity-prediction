#!/usr/bin/env bash
set -euo pipefail

# Optional dry-run mode (-v)
DRY_RUN=false
while getopts "v" opt; do
  case $opt in
    v) DRY_RUN=true ;;
    *) echo "Usage: $0 [-v]" >&2; exit 1 ;;
  esac
done

run_cmd() {
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    "$@"
  fi
}

split_types=("pocket" "scaffold")
folds=(0 1 2 3 4)

for split_type in "${split_types[@]}"; do
  for fold in "${folds[@]}"; do
    cgnn_name="cgnn_${split_type}_${fold}"
    cgnn3d_name="cgnn3d_${split_type}_${fold}"
    dimenet_name="dimenet_${split_type}_${fold}"

    run_cmd ./pli_alignment_scripts/explain_model.sh cgnn "$cgnn_name"
    run_cmd ./pli_alignment_scripts/explain_model.sh cgnn3d "$cgnn3d_name"
    run_cmd ./pli_alignment_scripts/explain_model.sh dimenet "$dimenet_name"
  done
done