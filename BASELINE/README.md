# AI-HPC Lab - BASELINE

Antoine Metz @iNeKkaa | Thierry Berard @ThryB64
## Setup

The baseline was run on FT3 using one NVIDIA A100 GPU.

Main software versions:

- Python 3.10.8
- PyTorch 2.5.1+cu121
- Transformers 4.57.1
- Datasets 4.3.0
- Accelerate 1.11.0
- Evaluate 0.4.6
- TensorBoard 2.21.0

The environment can be created with:

```bash
module purge
module load python/3.10.8

python -m venv $STORE/aihpc_venv
source $STORE/aihpc_venv/bin/activate

python -m pip install -r requirements.txt
```

## Implementation

The training code is based on the Hugging Face PyTorch Question Answering example.

Model:

```text
google-bert/bert-base-uncased
```

Dataset:

```text
SQuAD 1.1
```

The Hugging Face scripts handle the tokenization of SQuAD and the start/end positions used for Question Answering.

## Run

The baseline is submitted with:

```bash
sbatch baseline.slurm
```

The SLURM script requests:

```text
1 x NVIDIA A100
32 CPUs
32 GB RAM
```

The training command and parameters are defined in `run_baseline.sh`.

## Configuration

For the baseline benchmark, I used a subset of SQuAD so that the same workload can later be used for the distributed tests.

| Parameter | Value |
|---|---:|
| Training samples | 10,000 |
| Validation samples | 500 |
| Epochs | 4 |
| Batch size | 12 |
| Learning rate | 3e-5 |
| Max sequence length | 384 |
| Document stride | 128 |
| Precision | FP32 |
| GPU | 1 x A100 |

## Results

SLURM job: `10022270`

| Metric | Result |
|---|---:|
| Training time | 649.19 s |
| Total wall-clock time | 698 s |
| Throughput | 61.616 samples/s |
| Steps/s | 5.139 |
| Train loss | 0.9136 |
| Exact Match | 74.0 |
| F1 | 78.552 |
| GPU peak memory delta | ~2820 MB |

TensorBoard logging was enabled during the training. More detailed profiling will be included in the distributed part of the project.

The training loss decreased during the four epochs, from values above 3 at the beginning to around 0.3 near the end.

## Additional full SQuAD run

I also tested the same pipeline with the complete SQuAD training dataset.

- 87,599 original training examples
- 88,524 training features after tokenization
- 10,570 validation examples
- 2 epochs
- batch size 12
- 1 x A100

SLURM job: `9974405`

| Metric | Result |
|---|---:|
| Training time | 2865.92 s |
| Total wall-clock time | 3024 s |
| Throughput | 61.777 samples/s |
| Exact Match | 81.0785 |
| F1 | 88.3867 |

The throughput is almost the same as in the 10k run. For this reason, the 10k configuration is kept as the reference workload for the distributed experiments.

## Files

```text
BASELINE/
├── hf_qa/
│   ├── run_qa.py
│   ├── trainer_qa.py
│   └── utils_qa.py
├── logs/
│   ├── baseline_10022270.out
│   ├── baseline_10022270.err
│   ├── baseline_9974405.out
│   └── baseline_9974405.err
├── baseline.slurm
├── run_baseline.sh
├── requirements.txt
├── requirements-lock.txt
└── README.md
```
