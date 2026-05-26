# create output directory
mkdir -p txt_output_files

ROOT_PATH="./"
DATA_PATH='./'

SEED=2020
NITERS=50000
BATCH_SIZE=32
LR=1e-4

# For training latent diffusion model with continuous conditional labels (session type and frames)
echo "-------------------------------------------------------------------------------------------------"
echo "Continuous conditional latent diffusion model"
python main.py --root_path $ROOT_PATH --data_path $DATA_PATH --LDM CcLDM --seed $SEED --niters $NITERS --resume_niters 0 --save_niters_freq 2000 --lr $LR --batch_size $BATCH_SIZE 2>&1 | tee txt_output_files/output_CcLDM.txt
