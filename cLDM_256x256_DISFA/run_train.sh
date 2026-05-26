#!/bin/bash

ROOT_PATH="./"
DATA_PATH="./"

SEED=2020
NITERS=100000
BATCH_SIZE=64
LR=1e-4

# Create output directory if it doesn't exist
mkdir -p txt_output_files

# For training cLDM 
echo "-------------------------------------------------------------------------------------------------"
echo "Training conditional Latent Diffusion Model (cLDM) for DISFA dataset"
python main.py \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --LDM CcLDM \
  --seed "$SEED" \
  --niters "$NITERS" \
  --resume_niters 0 \
  --save_niters_freq 2000 \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee txt_output_files/output_cLDM.txt
