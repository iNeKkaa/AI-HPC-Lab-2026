#!/bin/bash
#SBATCH --job-name=bert_ddp_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=ddp_2gpu_%j.out
#SBATCH --error=ddp_2gpu_%j.err

module purge
module load python/3.10.8

source "$STORE/aihpc_venv/bin/activate"
export HF_HOME="$STORE/hf_cache"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "SLURM job: $SLURM_JOB_ID"

nvidia-smi

./run_ddp_2gpu.sh
