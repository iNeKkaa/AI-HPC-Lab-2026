# AI-HPC Lab - DISTRIBUTED

Antoine Metz @iNeKkaa | Thierry Berard @ThryB64

Distributed implementation of the BERT/SQuAD baseline.

Execution details, benchmarks and profiling results will be added as the distributed experiments are completed.

## 2-GPU DDP test

The first distributed version was tested on one node with two NVIDIA A100 GPUs using PyTorch DDP.

The per-GPU batch size was set to 6, keeping the same global batch size of 12 as the baseline.

Reference run:

- SLURM job: 10833716
- Training time: 419.01 s
- Total wall-clock time: 563 s
- Throughput: 95.463 samples/s
- Exact Match: 76.4
- F1: 80.7627
