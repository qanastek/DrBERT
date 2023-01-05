#!/bin/bash
#SBATCH --job-name=mlm_cont     # nom du job
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=128                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=mlm_test%j.out # nom du fichier de sortie
#SBATCH --error=mlm_test%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
#
# Envoi des mails
#SBATCH --mail-type=begin,fail,abort,end
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Chargement des modules
module load pytorch-gpu/py3/1.12.1
 
# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=10

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python -u run_train_jeanzay.py \
    --model_name_or_path='../data/camembert_base_hf' \
    --tokenizer_name='../data/camembert_base_hf' \
    --path_load_dataset="../data/tokenized_dataset" \
    --output_dir='./Model_out/' \
    --logging_dir='./Model_out/logs/' \
    --per_device_train_batch_size=32 \
    --do_train \
    --warmup_steps=10000 \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --logging_steps=500 \
    --report_to='tensorboard' \
    --save_strategy='epoch' \
    --skip_memory_metrics='False' \
    --log_level='info' \
    --logging_first_step='True' \
    --num_train_epochs=400 \
    --fp16 \
    --save_total_limit=400 \
    --ddp_timeout=600 \
    --ddp_find_unused_parameters='False' \
    

