#!/bin/bash
#SBATCH --job-name=bert_fsdp_4gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=fsdp_4gpu_%j.out
#SBATCH --error=fsdp_4gpu_%j.err

module purge
module load python/3.10.8

source "$STORE/aihpc_venv/bin/activate"
export HF_HOME="$STORE/hf_cache"

cd "$SLURM_SUBMIT_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

export MASTER_ADDR
export MASTER_PORT

echo "SLURM job: $SLURM_JOB_ID"
echo "Nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

srun bash -c '
    export NODE_RANK=$SLURM_NODEID
    ./run_fsdp_4gpu.sh
'
