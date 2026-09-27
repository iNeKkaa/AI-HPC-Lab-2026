#!/bin/bash
set -e

export HF_HOME="$STORE/hf_cache"

MODEL="google-bert/bert-base-uncased"
OUTPUT_DIR="$STORE/aihpc_outputs/baseline"

mkdir -p "$OUTPUT_DIR"

echo "AI-HPC BASELINE"
echo "Model: $MODEL"
echo "Dataset: SQuAD 1.1"
echo "GPU count requested: 1"
echo "Precision: FP32"
echo "Start: $(date)"

START_TIME=$(date +%s)

python hf_qa/run_qa.py \
  --model_name_or_path "$MODEL" \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --save_strategy no \
  --logging_steps 100 \
  --skip_memory_metrics False \
  --report_to tensorboard

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "End: $(date)"
echo "Total wall-clock time: ${ELAPSED} seconds"
echo "Total wall-clock time: $((ELAPSED / 60)) min $((ELAPSED % 60)) s"
