#!/bin/bash
set -e

export HF_HOME="$STORE/hf_cache"

MODEL="google-bert/bert-base-uncased"
OUTPUT_DIR="$STORE/aihpc_outputs/baseline_10k"

mkdir -p "$OUTPUT_DIR"

echo "AI-HPC BASELINE"
echo "Model: $MODEL"
echo "Dataset: SQuAD 1.1"
echo "Training subset: 10,000 examples"
echo "Validation subset: 500 examples"
echo "Epochs: 4"
echo "Batch size: 12"
echo "GPU count requested: 1"
echo "Precision: FP32"
echo "Start: $(date)"

START_TIME=$(date +%s)

python hf_qa/run_qa.py \
  --model_name_or_path "$MODEL" \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --max_train_samples 10000 \
  --max_eval_samples 500 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
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
