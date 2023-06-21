#!/bin/bash
#SBATCH --job-name=DrBERT
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=19:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH -A rtl@v100

module purge
module load pytorch-gpu/py3/1.11.0

nvidia-smi

srun python preprocessing_dataset.py \
    --model_type='camembert' \
    --tokenizer_name='./Tokenizer/' \
    --train_file='../data/corpus.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="../data/tokenized_dataset" \
    --output_dir='./test' \
    --preprocessing_num_workers=20


