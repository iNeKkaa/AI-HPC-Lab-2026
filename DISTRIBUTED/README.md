# AI-HPC Lab - DISTRIBUTED

Antoine Metz (@iNeKkaa)
Thierry Berard (@ThryB64)

Distributed version of the BERT/SQuAD baseline using PyTorch on FT3.

## Setup

All experiments use the same training workload:

- BERT base uncased
- SQuAD 1.1
- 10,000 training examples
- 500 validation examples
- 4 epochs
- FP32
- global batch size: 12

To keep the comparison fair, the global batch size was kept constant:

- 1 GPU: batch size 12
- 2 GPUs: batch size 6 per GPU
- 4 GPUs: batch size 3 per GPU

As the effective batch size does not change, the same learning rate of `3e-5` is used for all configurations.

## DDP

DistributedDataParallel was tested first.

The 2-GPU experiment runs on one FT3 node. The 4-GPU experiment uses two nodes with two A100 GPUs per node.

SLURM is used to allocate the resources and `torchrun` starts one process per GPU.

Each distributed configuration was executed three times.

## Scaling results

The baseline uses one A100 and the distributed results use the mean throughput of the three runs.

| GPUs | Strategy | samples/s | samples/s/GPU | Efficiency | Mean final loss |
|---:|---|---:|---:|---:|---:|
| 1 | Baseline | 61.616 | 61.616 | 100.0% | 0.9136 |
| 2 | DDP | 96.562 | 48.281 | 78.4% | 0.9045 |
| 4 | DDP | 125.264 | 31.316 | 50.8% | 0.9119 |
| 2 | FSDP | 85.880 | 42.940 | 69.7% | 0.8999 |
| 4 | FSDP | 98.962 | 24.740 | 40.2% | 0.9149 |

The median training times were:

| Strategy | GPUs | Median training time | Speedup vs baseline |
|---|---:|---:|---:|
| Baseline | 1 | 649.19 s | 1.00x |
| DDP | 2 | 413.04 s | 1.57x |
| DDP | 4 | 317.42 s | 2.05x |
| FSDP | 2 | 461.72 s | 1.41x |
| FSDP | 4 | 402.06 s | 1.61x |

DDP scales well from one to two GPUs, but the efficiency decreases with four GPUs.

The 4-GPU configuration runs across two nodes and the batch per GPU is smaller. Communication therefore represents a larger part of each training step.

The Exact Match and F1 results stayed in a similar range across the experiments, so the faster configurations did not produce a clear loss of model quality.

Raw results are available in:

- `results/benchmark_runs.csv`
- `results/scaling.csv`

## FSDP

FullyShardedDataParallel was also tested with `full_shard` and automatic wrapping of BERT layers.

For BERT-base, FSDP was slower than DDP but required less GPU memory.

| Strategy | GPUs | GPU allocated memory |
|---|---:|---:|
| DDP | 2 | ~1281 MB |
| FSDP | 2 | ~860 MB |
| DDP | 4 | ~1266 MB |
| FSDP | 4 | ~438 MB |

On two GPUs, FSDP reduced the measured allocated GPU memory by about 33%.

On four GPUs, the reduction was about 65%.

BERT-base already fits comfortably on an A100, so DDP gives better throughput for this workload. FSDP becomes more interesting when reducing memory usage is more important than maximizing throughput.

## Profiling

A short 2-GPU DDP run was analyzed with `torch.profiler`.

The profile showed that:

- `aten::mm` represented about 35% of CUDA time
- NCCL all-reduce operations represented about 27% of CUDA time

The all-reduce cost comes from synchronizing gradients between the DDP processes and helps explain why the scaling is not linear when more GPUs are added.

The profiling run was only used for analysis and is not included in the benchmark timings.

The profiler summary is available in:

`results/profile_summary.txt`

## Main files

- `run_ddp_2gpu.sh`: DDP training on 2 GPUs
- `ddp_2gpu.slurm`: SLURM script for 2-GPU DDP
- `run_ddp_4gpu.sh`: DDP training on 4 GPUs and 2 nodes
- `ddp_4gpu.slurm`: SLURM script for multi-node DDP
- `run_fsdp_2gpu.sh`: FSDP training on 2 GPUs
- `fsdp_2gpu.slurm`: SLURM script for 2-GPU FSDP
- `run_fsdp_4gpu.sh`: FSDP training on 4 GPUs and 2 nodes
- `fsdp_4gpu.slurm`: SLURM script for multi-node FSDP
- `fsdp_config.json`: FSDP configuration
- `run_profile_2gpu.sh`: short PyTorch profiler run
- `profile_2gpu.slurm`: SLURM profiling script
- `hf_qa/`: Hugging Face question answering scripts
- `logs/`: SLURM output files
- `results/`: benchmark and profiling results

## Running

2-GPU DDP:

    sbatch ddp_2gpu.slurm

4-GPU multi-node DDP:

    sbatch ddp_4gpu.slurm

2-GPU FSDP:

    sbatch fsdp_2gpu.slurm

4-GPU multi-node FSDP:

    sbatch fsdp_4gpu.slurm
