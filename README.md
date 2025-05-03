# Large-Scale-AI-Engineering-Project

This repo contains the final project for the "Large Scale AI Engineering" course at ETH Zürich, FS2025. The goal is to implement *Pretokenization* and investigate its effect on training LLMs on large GPU clusters (the Alps cluster at CSCS).

## Project Overview

Training large language models involves converting text to tokens before feeding them to the model. This tokenization process is typically done on-the-fly during training, which can become a bottleneck. This project explores how *pretokenization* - processing the text into tokens beforehand - can improve training efficiency.

## Baselines

We have two baseline implementations to compare against:

1. **Padded Dataset** (`src/data/dataset.py` using `ParquetDataset` class):
   - Straight-forward approach that pads all sequences to the same length
   - Around 75% of tokens are padding tokens that don't contribute to learning
   - Baseline throughput: ~6,544 tokens/sec

2. **Padding-free Dataset** (`src/data/iterable_dataset.py` using `IterableParquetDataset` class):
   - Advanced approach that eliminates padding by creating a continuous stream of tokens
   - Only 0.15% of tokens are ignored in loss calculation
   - Baseline throughput: ~6,903 tokens/sec

---

# Pretokenized Padded Dataset Implementation

We implemented a preprocessing-based tokenization approach that moves the tokenization process out of the training loop into a one-time preprocessing stage.

## Padded Parquet Pretokenization (Version 1)

### Storage Format
- **Implementation**: `src/preprocessing/pretokenize.py` processes text documents in batches
- **Format**: Parquet file with two columns: `input_ids` (token IDs) and `attention_mask` (padding indicators)
- **Compression**: ZSTD level 22 compression reduces the data size from 2.3GB (raw text) to 1.0GB (tokenized), achieving ~2.3x compression
- **Metadata**: Embedded schema metadata tracks tokenizer name, format type, and creation timestamp

### Data Retrieval
- **Access Pattern**: `PreTokenizedDataset` in `src/data/pretokenized_dataset.py` uses PyArrow's memory-mapped reader
- **Integration**: Implements the PyTorch Dataset interface, compatible with existing collator
- **Validation**: Verifies file format metadata before loading to ensure compatibility

### Training Integration
- **Preprocessing Command**: Run via `scripts/preprocessing/run_preprocessing_pretokenize_padded.sh`
- **Training Usage**: Enable with `--pretokenized` flag in `src/train.py`
- **Complete Pipeline**: `scripts/run_pretokenized_padded.sh` demonstrates full setup

The benefits are, that we can still train with different batch sizes without reprocessing data. Furthermore, the tokenization performed once regardless of training iterations and it is seamlessly integrated with existing training code. 

However, our initial benchmarks show that the pretokenized approach achieves lower throughput (6308 tokens/sec) compared to both baseline approaches (6544 and 6903 tokens/sec). This counterintuitive result likely stems from several factors: the overhead of Parquet file access during training, decompression costs from the high ZSTD compression level, and potential I/O bottlenecks when reading from disk. 