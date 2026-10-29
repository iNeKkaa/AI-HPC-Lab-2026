#!/bin/bash
set -e

export HF_HOME="$STORE/hf_cache"

MODEL="google-bert/bert-base-uncased"
OUTPUT_DIR="$STORE/aihpc_outputs/fsdp_2gpu"

mkdir -p "$OUTPUT_DIR"

echo "AI-HPC DISTRIBUTED - FSDP"
echo "Model: $MODEL"
echo "Dataset: SQuAD 1.1"
echo "Training subset: 10,000 examples"
echo "Validation subset: 500 examples"
echo "Epochs: 4"
echo "GPUs: 2"
echo "Batch size per GPU: 6"
echo "Global batch size: 12"
echo "FSDP strategy: full_shard"
echo "Precision: FP32"
echo "Start: $(date)"

START_TIME=$(date +%s)

torchrun \
  --standalone \
  --nproc_per_node=2 \
  hf_qa/run_qa.py \
  --model_name_or_path "$MODEL" \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --max_train_samples 10000 \
  --max_eval_samples 500 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --save_strategy no \
  --logging_steps 100 \
  --skip_memory_metrics False \
  --report_to tensorboard \
  --ddp_backend nccl \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config fsdp_config.json

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "End: $(date)"
echo "Total wall-clock time: ${ELAPSED} seconds"
echo "Total wall-clock time: $((ELAPSED / 60)) min $((ELAPSED % 60)) s"
