#!/bin/sh

DATASET_PATH=../DATASET_Lungs
CHECKPOINT_PATH=../unetr_pp/evaluation/unetr_pp_lung_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_lung 6 0 -val
