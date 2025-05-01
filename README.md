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

2. **Streaming Dataset** (`src/data/iterable_dataset.py` using `IterableParquetDataset` class):
   - Advanced approach that eliminates padding by creating a continuous stream of tokens
   - Only 0.15% of tokens are ignored in loss calculation
   - Baseline throughput: ~6,903 tokens/sec