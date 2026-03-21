These scripts can be used to retrain and explain the models studied in [Evaluation Beyond Goodness of Fit: Quantifying Biophysical Alignment of AI Models for Kinase-Centric Drug Discovery](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-qsw7v-v3).

### Retraining a model
To retrain a model run
```
./fit_model.sh ${MODEL_SCRIPT} ${MODEL_ID} ${EXTRA_ARGS}
```
where `MODEL_SCRIPT` is one of `(train_dimenet, train_sparse_transformer)`, `MODEL_ID` will be used as a string identifier in naming the output files and `EXTRA_ARGS` are any additional arguments to be passed to the training script (e.g. `--num_epochs 100`, `--hidden_channels 128`, ...).

For example, to retrain DimeNet++ on the scaffold split, fold 0, run:
```
./pli_alignment_scripts/fit_model.sh train_dimenet dimenet_scaffold_0 --num_epochs 300 --hidden_channels 256  --lr 0.0001 --split_type scaffold-k-fold --split_index 0
```

This will create three files
- `dimenet_scaffold_0.csv` containing the predictions of the model on the train and test set
- `dimenet_scaffold_0.pth` containing the model checkpoint
- `dimenet_scaffold_0.json` containing the training configuration

The script `fit_paper_models.sh` generates all calls to `fit_model.sh` that are required to reproduce the models studied in the publication.


### Explaining a model

#### Data requirement: atom/residues mapping files
Under the hood, the masking procedure makes use of cached mapping files that map the node indices in our input graphs to a numerical index that identifies which amino acid residue they are part of.
If not already done, these files can be obtained [from Zenodo](https://zenodo.org/records/19145842).

#### Masked residue prediction
To explain a model (generate predictions on inputs with and without masked residues), we can run
```
./pli_alignment_scripts/explain_model.sh ${MODEL_TYPE} ${MODEL_ID}
```
where `MODEL_TYPE` is one of `(dimenet, cgnn, cgnn3d)` and `MODEL_ID` is the string identifier used when training the model.
For example, to explain the DimeNet++ model trained on the scaffold split, fold 0, run:
```
./pli_alignment_scripts/explain_model.sh dimenet dimenet_scaffold_0
```

This will create two files
- `dimenet_scaffold_0.masked.csv` containing the predictions of the model on the train and test set for the modified inputs where one single kinase residue is masked at a time (by removing the atoms/nodes from the original input graph).
- `dimenet_scaffold_0.ref.csv` containing the predictions of the model on the train and test set for the original, unmodified inputs.

These csv files can be processed further with the code in the `pliar` repository to obtain PLI-alignment R-AUROC scores.
Re-running all model training and evaluation is not strictly necessary. Instead, you can also obtain pre-computed model predictions (including masked predictions) [from Zenodo](https://zenodo.org/records/19145842).

The script `explain_paper_models.sh` generates all calls to `explain_model.sh` that are required to reproduce the explanations (masked predictions) for the models studied in the publication.

### Verbosity option for entry point output
In our experiments, both training and model evaluation where split across multiple jobs on an HTCondor cluster that executed on machines with NVIDIA A100 GPUs.
You can also use the verbosity `-v` option of both `train_paper_models.sh` and `explain_paper_models.sh` to print the commands that would be executed without actually running them (dry-run mode).
This enables you to reuse these entry points for execution on your compute infrastructure of choice (e.g. local machine, Slurm cluster, ...).