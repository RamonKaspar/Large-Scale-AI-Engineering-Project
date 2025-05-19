#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai_ddp_baseline_pretokenized_padded_ws_16_bs_8
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/logs/%x-%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --environment=/iopsstor/scratch/cscs/kasparr/project/ngc_pt_jan.toml

# Stop the script if a command fails
set -eo pipefail

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

# Environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
export PYTHONUNBUFFERED=1   # NOTE: For debugging to see the python output

echo "[Master] Master node: $MASTER_ADDR"
echo "[Master] World size: $WORLD_SIZE"

# Command to run on each node
CMD="
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

# Run the script
torchrun \
    --nnodes=${SLURM_NNODES} \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /iopsstor/scratch/cscs/$USER/project/src/train.py \
    --distributed \
    --dataset /capstor/scratch/cscs/kasparr/project/train_data_tokenized_padded_snappy.parquet \
    --dataset-type padded \
    --pretokenized \
    --sequence-length 2048 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --logging-frequency 10
"

# Run the command on all nodes
srun bash -c "$CMD"

echo "[sbatch-master] task finished"