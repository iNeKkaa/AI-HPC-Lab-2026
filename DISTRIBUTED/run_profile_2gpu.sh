#!/bin/bash
set -e

export HF_HOME="$STORE/hf_cache"
export PROFILER_DIR="$STORE/aihpc_profiles/ddp_2gpu"

MODEL="google-bert/bert-base-uncased"
OUTPUT_DIR="$STORE/aihpc_outputs/ddp_2gpu_profile"

rm -rf "$PROFILER_DIR"
mkdir -p "$PROFILER_DIR"
mkdir -p "$OUTPUT_DIR"

echo "AI-HPC DDP PROFILING"
echo "GPUs: 2"
echo "Batch size per GPU: 6"
echo "Profiler output: $PROFILER_DIR"
echo "Start: $(date)"

torchrun \
  --standalone \
  --nproc_per_node=2 \
  hf_qa/run_qa_profile.py \
  --model_name_or_path "$MODEL" \
  --dataset_name squad \
  --do_train \
  --max_train_samples 512 \
  --per_device_train_batch_size 6 \
  --learning_rate 3e-5 \
  --max_steps 8 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --save_strategy no \
  --logging_steps 1 \
  --report_to none \
  --ddp_backend nccl \
  --ddp_find_unused_parameters false

echo "End: $(date)"
