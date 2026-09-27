#!/bin/bash
#SBATCH --job-name=bert_baseline
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

module purge
module load python/3.10.8

source "$STORE/aihpc_venv/bin/activate"
export HF_HOME="$STORE/hf_cache"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "SLURM job: $SLURM_JOB_ID"

nvidia-smi

./run_baseline.sh
