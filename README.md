# Large-Scale-AI-Engineering-Project

This repo contains the final project for the "Large Scale AI Engineering" course at ETH Zürich, FS2025. The goal is to implement *Pretokenization* and investigate its effect on training LLMs on large GPU clusters. We conducted our experiments on the Alps cluster at CSCS, which features NVIDIA GH200 GPUs with 120GB of VRAM.

## Project Overview

Training large language models involves converting text to tokens before feeding them to the model. This tokenization process is typically done on-the-fly during training, which can become a bottleneck. This project explores if *pretokenization* - processing the text into tokens beforehand - can improve training efficiency.

## Baselines

We have two baseline implementations to compare against:

1. **Padded Dataset** (`src/data/dataset.py` using `ParquetDataset` class):
   - Straight-forward approach that pads all sequences to the same length
   - Around 75% of tokens are padding tokens that don't contribute to learning

2. **Padding-free Dataset** (`src/data/iterable_dataset.py` using `IterableParquetDataset` class):
   - Advanced approach that eliminates padding by creating a continuous stream of tokens
   - Only 0.15% of tokens are ignored in loss calculation

## Pretokenized Implementations

We implemented a preprocessing-based tokenization approach that moves the tokenization process out of the training loop into a one-time preprocessing stage.

Our pretokenized implementations were designed to map 1:1 to our baseline approaches, with minimal modifications to ensure the fairest possible comparison. We maintained the same token handling logic, masking approach, and sequence construction as the original implementations, only removing the on-the-fly tokenization step.

### Pretokenization Script

We created a dedicated preprocessing script (`src/preprocessing/pretokenize.py`) that converts text data into two formats:

- **Pretokenized Padded Format**: Which tokenizes text with padding to fixed sequence length, and stores input IDs and attention masks. This matches exactly the format expected by the original `ParquetDataset`

- **Pretokenized Token-List (padding-free) Format**: Which tokenizes text without padding, storing raw token IDs, and preserves the token stream for continuous processing. This mirrors the data expected by the original `IterableParquetDataset`.

We save both as a Parquet file with Snappy compression (same as the original one), with `row_group_size=1000` for efficient batch loading. With the script `analysis/pretokenization_verification.py` we verified the pretokenization and made sure it matches the on-the-fly one.

### Dataset Implementations & Training Adaption

We have to modify the dataloaders and training script to load directly the pretokenized data, without on-the-fly tokenization. These newly created  dataset classes can be found in `src/data/pretokenized_dataset.py`.

- **PreTokenizedDataset**: Equivalent to the original `ParquetDataset`; loads pretokenized padded input_ids and attention_masks directly

- **IterablePreTokenizedDataset**: Equivalent to the original `IterableParquetDataset`; streams pretokenized tokens without padding and preserves BOS token handling and loss masking logic.

The training script was updated with a `pretokenized` flag that allows seamless switching between on-the-fly tokenization and pretokenization, while keeping all other training parameters identical.

## How to Run

For running on the CSCS Alps cluster, see the SLURM job scripts in the `scripts` directory that configure the environment settings for all experiment configurations and can be started as `sbatch` jobs.

### Step 1: Pretokenize the Dataset

```bash
python pretokenize.py \
  --input-path /path/to/train_data.parquet \
  --output-dir /path/to/output \
  --tokenizer-name unsloth/Mistral-Nemo-Base-2407-bnb-4bit \
  --max-length 2048 \
  --batch-size 100 \
  --format padded token-list
```

This creates two output files:
- `train_data_tokenized_padded_snappy.parquet`
- `train_data_tokenized_token-list_snappy.parquet`

### Step 2: Train Using Pretokenized Data

```bash
python src/train.py \
  --dataset /path/to/train_data_tokenized_padded_snappy.parquet \
  --dataset-type padded \
  --pretokenized \
  --sequence-length 2048 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --lr-warmup-steps 100 \
  --training-steps 1000
```

For the token-list format:
```bash
python src/train.py \
  --dataset /path/to/train_data_tokenized_token-list_snappy.parquet \
  --dataset-type token-list \
  --pretokenized \
  --sequence-length 2048 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --lr-warmup-steps 100 \
  --training-steps 1000
```

---

## Performance Results

### Raw Data Processing Speed Comparison

We conducted isolated benchmarks to measure the raw speed advantage of pretokenization without the overhead of the full training pipeline. The benchmark processed 1000 samples using three approaches and measured both dataset loading and sample processing times:

![alt text](plots/tokenization_benchmark.png)

The results show that while on-the-fly tokenization processes each sample in 2.70 ms, pretokenized approaches offer significant speedups:
- Pretokenized padded: 1.68 ms/sample (1.61× faster)
- Pretokenized token-list: 0.46 ms/sample (5.9× faster)

However, as we will see in the next section, pretokenization in the training pipeline is not actually beneficial!

### Training Results

Our experiments comparing on-the-fly tokenization versus pretokenization showed:

| Configuration                           | Tokens/sec | MFU % | TFLOPs |
|----------------------------------------|------------|--------|--------|
| Baseline padded (batch_size=1)         | 6544.40    | 31.97  | 316.23 |
| Baseline padded (batch_size=2)         | 7873.05    | 38.47  | 380.43 |
| Baseline padding-free (batch_size=1)   | 6902.55    | 35.97  | 355.77 |
| Baseline padding-free (batch_size=2)   | 7410.96    | 36.21  | 358.10 |
| Pretokenized padded (batch_size=1)     | 6333.60    | 30.94  | 306.04 |
| Pretokenized padded (batch_size=2)     | 7605.06    | 37.16  | 367.48 |
| Pretokenized token-list (batch_size=1) | 6224.24    | 30.41  | 300.76 |
| Pretokenized token-list (batch_size=2) | 7475.30    | 36.52  | 361.21 |

All implementations showed improved performance when increasing batch size from 1 to 2, with better hardware utilization with larger batches. Memory constraints prevented testing with larger batch sizes. Furthermore, our experiments confirm pretokenization maintains similar learning behavior to on-the-fly tokenization, as shown by comparable validation loss curves -- so it doesn't negatively impact model quality or learning dynamics.

However, in conclusion we have to say that pretokenization is in this case not worth it. This might be due to the GPU compute being the main bottleneck rather than tokenization in our training setup. In the future, we definitely should evaluate performance using Distributed Data Parallel (DDP) training across multiple GPUs to enable larger effective batch sizes (e.g., 32 or higher).

![alt text](plots/batch_size_comparison.png)

![alt text](plots/loss_over_time.png)