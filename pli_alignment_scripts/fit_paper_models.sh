split_types = ("pocket" "scaffold")
folds = (0 1 2 3 4)

CGNNX_ARGS=
DIMENET_ARGS=

for split_type in "${split_types[@]}"; do
  for fold in "${folds[@]}"; do
    cgnn_name="cgnn_${split_type}_${fold}"
    cgnn3d_name="cgnn3d_${split_type}_${fold}"
    dimenet_name="dimenet_${split_type}_${fold}"

    ./fit_model.sh train_sparse_transformer $cgnn_name --split_type "${split_type}-k-fold" --split_index $fold --covalent_only True
    ./fit_model.sh train_sparse_transformer $cgnn3d_name --split_type "${split_type}-k-fold" --split_index $fold --covalent_only False
    ./fit_model.sh train_dimenet $dimenet_name --split_type "${split_type}-k-fold" --split_index $fold
  done
done