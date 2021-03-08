#!/bin/bash pred_hoi_dets_${SUBSET}_${MODEL_NUM}

# bash compute_map.sh

SUBSET="test"
HICO_EXP_DIR="${PWD}"
OUT_DIR="${HICO_EXP_DIR}/mAP/each_hoi_ap_data"
MAP_DIR="${HICO_EXP_DIR}/mAP"
PROC_DIR="data/hico/hico_processed/"

EXP_NAME="PD"
Thres="0.01"
MODES=("Known-Object" "Default")

MODEL_NUM="185000"
for MODE in "${MODES[@]}"
do
    PRED_HOI_DETS_HDF5="${HICO_EXP_DIR}/output/hico-det/${EXP_NAME}/pred_hdf5/pred_hoi_dets_${SUBSET}_${MODEL_NUM}.hdf5"
    MY_EXP="PD"
    File_Name="train_sample"
    cd eval
    python -m compute_map \
        --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
        --out_dir $OUT_DIR \
        --proc_dir $PROC_DIR \
        --subset $SUBSET  \
        --exp_name $MY_EXP \
        --file_name $File_Name \
        --model_num $MODEL_NUM \
        --mAP_dir $MAP_DIR \
        --mode $MODE \
        --thres $Thres
done

