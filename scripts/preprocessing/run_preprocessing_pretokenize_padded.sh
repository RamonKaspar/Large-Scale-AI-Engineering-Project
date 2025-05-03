#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai_preprocessing_pretokenizing_padded
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/kasparr/project/ngc_pt_jan.toml
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/project"

CMD="python3 $ASSIGNMENT_DIR/src/preprocessing/pretokenize.py \
    --input-path /capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet \
    --output-dir /capstor/scratch/cscs/kasparr/project \
    --tokenizer-name unsloth/Mistral-Nemo-Base-2407-bnb-4bit \
    --max-length 2048 \
    --format padded"

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD"

echo "END TIME: $(date)"