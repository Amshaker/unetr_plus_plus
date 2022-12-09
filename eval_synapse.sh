export CUDA_VISIBLE_DEVICES=6
export nnFormer_raw_data_base=/share/users/maaz/nnFormer/DATASET/nnFormer_raw
export nnFormer_preprocessed=/share/users/maaz/nnFormer/DATASET/nnFormer_raw/nnFormer_raw_data/Task02_Synapse


export RESULTS_FOLDER=/share/users/maaz/nnFormer/checkpoints/nnFormer_unetr_edgenext_ds_synapse_training_run_1
python nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_nnformer_synapse 2 0 -val


export RESULTS_FOLDER=/share/users/maaz/nnFormer/checkpoints/nnFormer_unetr_edgenext_ds_synapse_training_run_2
python nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_nnformer_synapse 2 0 -val


export RESULTS_FOLDER=/share/users/maaz/nnFormer/checkpoints/nnFormer_unetr_edgenext_ds_synapse_training_run_3
python nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_nnformer_synapse 2 0 -val