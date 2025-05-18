#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai_ddp_validation
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/logs/%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --environment=/iopsstor/scratch/cscs/kasparr/project/ngc_pt_jan.toml

# Stop script if a command fails
set -eo pipefail

echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

# Environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345
export WORLD_SIZE=$(( SLURM_NNODES * 4 )) # 4 GPUs per node
export PYTHONUNBUFFERED=1

echo "[Master] Master node: $MASTER_ADDR"
echo "[Master] World size: $WORLD_SIZE"

# Command to run on each node
CMD="
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

# Run the script with torchrun to handle DDP setup
torchrun \
    --nnodes=${SLURM_NNODES} \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /iopsstor/scratch/cscs/$USER/project/src/validate_ddp.py
"

# Run the command on all nodes
srun bash -c "$CMD"

# Print the verification result
if [ -f "verification_results/verification_summary.txt" ]; then
    echo "=== VERIFICATION RESULT ==="
    cat verification_results/verification_summary.txt
    echo "=========================="
else
    echo "ERROR: Verification summary not found!"
    exit 1
fi

echo "[sbatch-master] task finished"