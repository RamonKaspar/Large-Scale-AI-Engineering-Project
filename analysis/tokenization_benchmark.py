"""
(To be run on the login node)
Tokenization Benchmark: Measures and visualizes performance our baseline, pretokenized padded, and pretokenized token-list datasets.
"""

import time
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

ORIGINAL_DATA = "/capstor/scratch/cscs/kasparr/project/train_data.parquet"
PRETOK_PADDED = "/capstor/scratch/cscs/kasparr/project/train_data_tokenized_padded_snappy.parquet"
PRETOK_TOKEN_LIST = "/capstor/scratch/cscs/kasparr/project/train_data_tokenized_token-list_snappy.parquet"

NUM_SAMPLES = 1000
SEQUENCE_LENGTH = 2048
TOKENIZER_NAME = "unsloth/Mistral-Nemo-Base-2407-bnb-4bit"

def benchmark_original_data(file_path, tokenizer, seq_length, num_samples):
    print(f"\nBenchmarking original data loading & tokenization: {file_path}")
    
    # Load parquet file
    load_start = time.time()
    parquet_ds = pq.read_table(file_path, memory_map=True)
    load_time = time.time() - load_start
    print(f"  Parquet load time: {load_time:.4f} seconds")
    
    # Measure tokenization time
    tokenize_start = time.time()
    for i in range(num_samples):
        idx = i % len(parquet_ds)
        text = str(parquet_ds["text"][idx].as_py())
        tokens = tokenizer.encode_plus(
            text,
            max_length=seq_length + 1,
            padding='max_length',
            truncation=True
        )
        # Create tensors
        input_tensor = torch.tensor(tokens["input_ids"][:-1])
        label_tensor = torch.tensor(tokens["input_ids"][1:])
    
    tokenize_time = time.time() - tokenize_start
    print(f"  Tokenization time for {num_samples} samples: {tokenize_time:.4f} seconds")
    print(f"  Time per sample: {tokenize_time/num_samples*1000:.4f} ms")
    
    return load_time, tokenize_time

def benchmark_pretok_padded(file_path, num_samples):
    print(f"\nBenchmarking pretokenized padded data loading: {file_path}")
    
    # Load parquet file
    load_start = time.time()
    parquet_ds = pq.read_table(file_path, memory_map=True)
    load_time = time.time() - load_start
    print(f"  Parquet load time: {load_time:.4f} seconds")
    
    # Measure retrieval time
    retrieve_start = time.time()
    for i in range(num_samples):
        idx = i % len(parquet_ds)
        input_ids = parquet_ds["input_ids"][idx].as_py()
        attention_mask = parquet_ds["attention_mask"][idx].as_py()
        
        # Create tensors
        input_tensor = torch.tensor(input_ids[:-1])
        label_tensor = torch.tensor(input_ids[1:])
    
    retrieve_time = time.time() - retrieve_start
    print(f"  Retrieval time for {num_samples} samples: {retrieve_time:.4f} seconds")
    print(f"  Time per sample: {retrieve_time/num_samples*1000:.4f} ms")
    
    return load_time, retrieve_time

def benchmark_pretok_token_list(file_path, num_samples):
    print(f"\nBenchmarking pretokenized token-list data loading: {file_path}")
    
    # Load parquet file
    load_start = time.time()
    parquet_ds = pq.read_table(file_path, memory_map=True)
    load_time = time.time() - load_start
    print(f"  Parquet load time: {load_time:.4f} seconds")
    
    # Measure retrieval time
    retrieve_start = time.time()
    for i in range(num_samples):
        idx = i % len(parquet_ds)
        tokens = parquet_ds["tokens"][idx].as_py()
        
        # Create tensors
        if len(tokens) > 1:
            input_tensor = torch.tensor(tokens[:-1])
            label_tensor = torch.tensor(tokens[1:])
    
    retrieve_time = time.time() - retrieve_start
    print(f"  Retrieval time for {num_samples} samples: {retrieve_time:.4f} seconds")
    print(f"  Time per sample: {retrieve_time/num_samples*1000:.4f} ms")
    
    return load_time, retrieve_time

def visualize_results(results, num_samples, output_dir="plots"):
    """Create a stacked bar plot visualizing dataset load time and sample processing time."""    
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    
    method_labels = {
        "original": "On-the-fly\nTokenization",
        "pretok_padded": "Pretokenized\nPadded", 
        "pretok_token_list": "Pretokenized\nToken-List"
    }
    
    methods = list(results.keys())
    load_times = [results[method][0] for method in methods]
    process_times = [results[method][1] for method in methods]
    
    labels = [method_labels.get(method, method) for method in methods]
    
    plt.figure(figsize=(12, 8))
    bar_width = 0.6
    x = np.arange(len(methods))
    
    plt.bar(x, load_times, bar_width, label='Dataset Load Time', color='#4C72B0')
    plt.bar(x, process_times, bar_width, bottom=load_times, label='Sample Processing Time', color='#55A868')
    
    for i, method in enumerate(methods):
        plt.text(i, load_times[i]/2, f'{load_times[i]:.2f}s', ha='center', color='white', fontweight='bold')
        
        process_time = process_times[i]
        ms_per_sample = process_time / num_samples * 1000
        plt.text(i, load_times[i] + process_time/2, 
                f'{process_time:.2f}s', 
                ha='center', color='white', fontweight='bold')
    
    if "original" in results:
        baseline_time = results["original"][1]  # Process time of original method
        for i, method in enumerate(methods):
            if method != "original":
                speedup = baseline_time / results[method][1]
                plt.text(i, results[method][0] + results[method][1] + 0.2, 
                         f'{speedup:.2f}x faster', ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title(f'Tokenization Performance: Load and Process {num_samples} Samples', fontsize=16)
    plt.xticks(x, labels, fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    max_height = max([lt + pt for lt, pt in zip(load_times, process_times)])
    plt.ylim(0, max_height * 1.25)  # Add 25% padding at the top
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tokenization_benchmark.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nBenchmark visualization created at {output_path}")


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Run benchmarks
    print(f"Running benchmarks with {NUM_SAMPLES} samples each...")
    
    results = {}
    
    # Original data
    results["original"] = benchmark_original_data(
        ORIGINAL_DATA, tokenizer, SEQUENCE_LENGTH, NUM_SAMPLES
    )
    
    # Pretokenized padded
    results["pretok_padded"] = benchmark_pretok_padded(
        PRETOK_PADDED, NUM_SAMPLES
    )
    
    # Pretokenized token-list
    results["pretok_token_list"] = benchmark_pretok_token_list(
        PRETOK_TOKEN_LIST, NUM_SAMPLES
    )
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Samples processed: {NUM_SAMPLES}")
    
    print("\nFile loading times:")
    for name, (load_time, _) in results.items():
        print(f"  {name.ljust(20)}: {load_time:.4f} seconds")
    
    print("\nProcessing times per sample:")
    for name, (_, process_time) in results.items():
        ms_per_sample = process_time/NUM_SAMPLES*1000
        print(f"  {name.ljust(20)}: {ms_per_sample:.4f} ms/sample")
    
    print("\nSpeed comparison (relative to original):")
    orig_process = results["original"][1]
    for name, (_, process_time) in results.items():
        if name != "original":
            speedup = orig_process/process_time
            print(f"  {name.ljust(20)} is {speedup:.2f}x faster")
    
    print("\nTotal time (load + process):")
    for name, (load_time, process_time) in results.items():
        total = load_time + process_time
        print(f"  {name.ljust(20)}: {total:.4f} seconds")
    
    visualize_results(results, NUM_SAMPLES)

if __name__ == "__main__":
    main()