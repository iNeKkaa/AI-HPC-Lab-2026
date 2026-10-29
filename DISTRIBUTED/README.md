# AI-HPC Lab - DISTRIBUTED

Antoine Metz (@iNeKkaa)
Thierry Berard (@ThryB64)

Distributed version of the BERT/SQuAD baseline using PyTorch distributed training on FT3.

## Setup

The experiments use:

- BERT base uncased
- SQuAD 1.1
- 10,000 training examples
- 500 validation examples
- 4 epochs
- FP32
- global batch size of 12

The global batch size was kept constant for the different GPU configurations:

- 1 GPU: batch size 12
- 2 GPUs: batch size 6 per GPU
- 4 GPUs: batch size 3 per GPU

## Distributed strategies

Two PyTorch distributed strategies were tested.

### DDP

DistributedDataParallel keeps a copy of the model on each GPU and synchronizes gradients between processes.

Tests were done with:

- 2 GPUs on 1 node
- 4 GPUs on 2 nodes

### FSDP

FullyShardedDataParallel was also tested using `full_shard` and automatic wrapping of BERT layers.

FSDP reduces the amount of model state stored on each GPU, at the cost of additional communication.

## Results

Each distributed configuration was executed three times. The median training time is used for the speedup calculation.

| Strategy | GPUs | Median training time | Mean throughput | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 649.19 s | 61.616 samples/s | 1.00x | 100% |
| DDP | 2 | 413.04 s | 96.562 samples/s | 1.57x | 78.6% |
| DDP | 4 | 317.42 s | 125.264 samples/s | 2.05x | 51.1% |
| FSDP | 2 | 461.72 s | 85.880 samples/s | 1.41x | 70.3% |
| FSDP | 4 | 402.06 s | 98.962 samples/s | 1.61x | 40.4% |

DDP gave the best training time for this model. Going from one to four GPUs reduced the training time from about 649 s to 317 s.

The scaling is not linear because communication becomes more important when more GPUs are used, especially across two nodes.

FSDP was slower than DDP for BERT-base, but reduced the memory allocated per GPU. The measured allocation went from about 1281 MB to 860 MB on 2 GPUs, and from about 1266 MB to about 438 MB on 4 GPUs.

For this model, which already fits easily on an A100, DDP is therefore more efficient for training speed while FSDP is more interesting for reducing memory usage.

The Exact Match and F1 scores remained in a similar range for all configurations.

Raw benchmark results are available in:

- `results/benchmark_runs.csv`
- `results/scaling.csv`

## Profiling

A short 2-GPU DDP run was also analyzed with `torch.profiler`.

The profile showed that matrix multiplications (`aten::mm`) represented about 35% of CUDA time, while NCCL all-reduce communication represented about 27%.

This helps explain the loss of parallel efficiency when increasing the number of GPUs.

The profiling run was only used for performance analysis and is not included in the benchmark timings.

The profiler summary is available in:

`results/profile_summary.txt`

## Main files

- `run_ddp_2gpu.sh`: DDP training on 2 GPUs
- `ddp_2gpu.slurm`: SLURM script for 2-GPU DDP
- `run_ddp_4gpu.sh`: multi-node DDP training
- `ddp_4gpu.slurm`: SLURM script for 4-GPU DDP
- `run_fsdp_2gpu.sh`: FSDP training on 2 GPUs
- `fsdp_2gpu.slurm`: SLURM script for 2-GPU FSDP
- `run_fsdp_4gpu.sh`: multi-node FSDP training
- `fsdp_4gpu.slurm`: SLURM script for 4-GPU FSDP
- `run_profile_2gpu.sh`: short profiling run
- `profile_2gpu.slurm`: SLURM profiling job
- `hf_qa/`: Hugging Face question answering scripts
- `logs/`: SLURM outputs
- `results/`: benchmark and profiling results

## Running

Example for 2-GPU DDP:

    sbatch ddp_2gpu.slurm

Example for multi-node DDP:

    sbatch ddp_4gpu.slurm

FSDP experiments can be launched in the same way with the corresponding SLURM scripts.
