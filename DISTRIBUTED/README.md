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

## Scaling results

The same training workload was used for all configurations. The global batch size was kept equal to 12:

- 1 GPU: batch size 12
- 2 GPUs: batch size 6 per GPU
- 4 GPUs: batch size 3 per GPU

| GPUs | Nodes | Training time | Throughput | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 649.19 s | 61.616 samples/s | 1.00x | 100% |
| 2 | 1 | 419.01 s | 95.463 samples/s | 1.55x | 77.5% |
| 4 | 2 | 325.22 s | 122.992 samples/s | 2.00x | 49.9% |

The 2-GPU configuration gives a clear reduction in training time. With four GPUs, the training is approximately twice as fast as the single-GPU baseline, although the scaling is not linear because DDP adds gradient synchronization and inter-node communication overhead.
