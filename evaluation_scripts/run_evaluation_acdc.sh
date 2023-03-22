#!/bin/sh

DATASET_PATH=../DATASET_Acdc
CHECKPOINT_PATH=../unetr_pp/evaluation/unetr_pp_acdc_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_acdc 1 0 -val 
