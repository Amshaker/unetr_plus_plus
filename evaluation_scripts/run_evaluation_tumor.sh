#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../unetr_pp/evaluation/unetr_pp_tumor_checkpoint
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw


# Only for Tumor, it is recommended to train unetr_plus_plus first, and then use the provided checkpoint to evaluate. It might raise issues regarding the pickle files if you evaluated without training

python ../unetr_pp/inference/predict_simple.py -i ../unetr_plus_plus/DATASET_Tumor/unetr_pp_raw/unetr_pp_raw_data/Task003_tumor/imagesTs -o ../unetr_plus_plus/unetr_pp/evaluation/unetr_pp_tumor_checkpoint/inferTs -m 3d_fullres  -t 3 -f 0 -chk model_final_checkpoint -tr unetr_pp_trainer_tumor


python ../unetr_pp/inference_tumor.py 0

