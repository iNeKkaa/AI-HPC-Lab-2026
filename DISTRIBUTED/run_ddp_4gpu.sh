#!/bin/bash
set -e

export HF_HOME="$STORE/hf_cache"

MODEL="google-bert/bert-base-uncased"
OUTPUT_DIR="$STORE/aihpc_outputs/ddp_4gpu"

mkdir -p "$OUTPUT_DIR"

if [ "$NODE_RANK" -eq 0 ]; then
    echo "AI-HPC DISTRIBUTED - DDP MULTI-NODE"
    echo "Model: $MODEL"
    echo "Dataset: SQuAD 1.1"
    echo "Training subset: 10,000 examples"
    echo "Validation subset: 500 examples"
    echo "Epochs: 4"
    echo "Nodes: 2"
    echo "GPUs per node: 2"
    echo "Total GPUs: 4"
    echo "Batch size per GPU: 3"
    echo "Global batch size: 12"
    echo "Precision: FP32"
    echo "Master: $MASTER_ADDR:$MASTER_PORT"
    echo "Start: $(date)"

    START_TIME=$(date +%s)
fi

echo "Node rank $NODE_RANK running on $(hostname)"

torchrun \
  --nnodes=2 \
  --nproc_per_node=2 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  hf_qa/run_qa.py \
  --model_name_or_path "$MODEL" \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --max_train_samples 10000 \
  --max_eval_samples 500 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 3 \
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
  --ddp_find_unused_parameters false

if [ "$NODE_RANK" -eq 0 ]; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "End: $(date)"
    echo "Total wall-clock time: ${ELAPSED} seconds"
    echo "Total wall-clock time: $((ELAPSED / 60)) min $((ELAPSED % 60)) s"
fi
