"""
Benchmarking script for comparing dataset implementations.
"""

import time
from typing import Dict, Any

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ParquetDataset, CollatorForCLM
from src.data.iterable_dataset import IterableParquetDataset


def benchmark_padded_dataset(dataset_path: str, tokenizer: Any, sequence_length: int, batch_size: int, num_samples: int) -> Dict[str, Any]:
    """Benchmark the padded ParquetDataset implementation."""
    print("\n=== Benchmarking ParquetDataset with padding ===")
    # Measure dataset initialization time
    start_time = time.time()
    dataset = ParquetDataset(
        parquet_file=dataset_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        training_samples=num_samples,
    )
    init_time = time.time() - start_time
    print(f"Dataset initialization time: {init_time:.4f} seconds")
    
    # Measure time to get first item
    start_time = time.time()
    sample = dataset[0]
    first_item_time = time.time() - start_time
    print(f"Time to get first item: {first_item_time:.4f} seconds")
    
    collator = CollatorForCLM(sequence_length=sequence_length, pad_token_id=tokenizer.pad_token_id)
    
    # Measure dataloader iteration time
    start_time = time.time()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    for batch_inputs, batch_labels in dataloader:
        batch_time = time.time() - start_time
        # Print shapes
        # print(f"Input shape: {batch_inputs.shape}")
        # print(f"Labels shape: {batch_labels.shape}")
        # Count ignored tokens in the loss calculation
        ignored_count = (batch_labels == -100).sum().item()
        total_label_tokens = batch_labels.numel()
        padding_percentage = ignored_count / total_label_tokens * 100
        print(f"Ignored tokens in loss: {ignored_count:,} out of {total_label_tokens:,} ({padding_percentage:.2f}%)")
        print(f"Time to get first batch: {batch_time:.4f} seconds")
        # Calculate effective tokens (non-padding)
        effective_tokens = total_label_tokens - ignored_count
        print(f"Effective tokens per batch: {effective_tokens:,} ({100-padding_percentage:.2f}% efficiency)")
        break
    return {
        "name": "Padded Dataset",
        "init_time": init_time,
        "first_item_time": first_item_time,
        "batch_time": batch_time,
        "padding_percentage": padding_percentage,
        "effective_tokens": effective_tokens,
        "total_tokens": total_label_tokens,
    }


def benchmark_padding_free_dataset(dataset_path: str, tokenizer: Any, sequence_length: int, batch_size: int) -> Dict[str, Any]:
    """Benchmark the padding-free IterableParquetDataset implementation."""
    print("\n=== Benchmarking IterableParquetDataset (padding-free) ===")
    # Measure dataset initialization time
    start_time = time.time()
    dataset = IterableParquetDataset(
        parquet_file=dataset_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None else 1
    )
    init_time = time.time() - start_time
    print(f"Dataset initialization time: {init_time:.4f} seconds")
    # Measure dataloader iteration time
    start_time = time.time()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch_inputs, batch_labels in dataloader:
        batch_time = time.time() - start_time
        # Print shapes
        # print(f"Input shape: {batch_inputs.shape}")
        # print(f"Labels shape: {batch_labels.shape}")
        # Count ignored tokens in the loss calculation
        ignored_count = (batch_labels == -100).sum().item()
        total_label_tokens = batch_labels.numel()
        padding_percentage = ignored_count / total_label_tokens * 100
        print(f"Ignored tokens in loss: {ignored_count:,} out of {total_label_tokens:,} ({padding_percentage:.2f}%)")
        print(f"Time to get first batch: {batch_time:.4f} seconds")
        # Calculate effective tokens (non-padding)
        effective_tokens = total_label_tokens - ignored_count
        print(f"Effective tokens per batch: {effective_tokens:,} ({100-padding_percentage:.2f}% efficiency)")
        break
    return {
        "name": "Padding-free Dataset",
        "init_time": init_time,
        "batch_time": batch_time,
        "padding_percentage": padding_percentage,
        "effective_tokens": effective_tokens,
        "total_tokens": total_label_tokens,
    }


def measure_tokenization_speed(tokenizer: Any, sequence_length: int = 1000):
    """Measure the raw tokenization speed of the tokenizer."""
    print("\n=== Measuring Raw Tokenization Speed ===")
    # Sample text (300 words)
    sample_text = """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.  
Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.  
Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.  
Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem"""
    
    # Measure encode time
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        tokens = tokenizer.encode(sample_text, max_length=sequence_length, truncation=True)
    encode_time = time.time() - start_time
    token_count = len(tokens)
    tokens_per_second = (token_count * num_iterations) / encode_time
    print(f"Sample text length: {len(sample_text)} characters")
    print(f"Tokenized to {token_count} tokens")
    print(f"Tokenization speed: {tokens_per_second:.2f} tokens/second (averaged over {num_iterations} runs)")
    return {
        "tokens_per_second": tokens_per_second,
        "token_count": token_count,
        "encode_time": encode_time,
    }


def benchmark_datasets(dataset_path: str, tokenizer_name: str, sequence_length: int = 4096, batch_size: int = 32, num_samples: int = 32):
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Measure raw tokenization speed
    tokenization_metrics = measure_tokenization_speed(tokenizer)
    # Benchmark padded dataset
    padded_metrics = benchmark_padded_dataset(
        dataset_path, 
        tokenizer, 
        sequence_length, 
        batch_size,
        num_samples,
    )
    # Benchmark padding-free dataset
    padding_free_metrics = benchmark_padding_free_dataset(
        dataset_path, 
        tokenizer, 
        sequence_length, 
        batch_size,
    )
    # Print comparative summary
    print("\n=== Benchmark Summary ===")
    print(f"{'Metric':<25} {'Padded':<15} {'Padding-free':<15}")
    print("-" * 60)
    print(f"{'Dataset init time (s)':<25} {padded_metrics['init_time']:<15.4f} {padding_free_metrics['init_time']:<15.4f}")
    print(f"{'Batch fetch time (s)':<25} {padded_metrics['batch_time']:<15.4f} {padding_free_metrics['batch_time']:<15.4f}")
    print(f"{'Padding percentage (%)':<25} {padded_metrics['padding_percentage']:<15.2f} {padding_free_metrics['padding_percentage']:<15.2f}")
    print(f"{'Effective tokens':<25} {padded_metrics['effective_tokens']:<15,d} {padding_free_metrics['effective_tokens']:<15,d}")
    print(f"{'Efficiency (%)':<25} {100-padded_metrics['padding_percentage']:<15.2f} {100-padding_free_metrics['padding_percentage']:<15.2f}")
    print("\nRaw Tokenization Speed:")
    print(f"Tokenizer processes approximately {tokenization_metrics['tokens_per_second']:.2f} tokens/second")
    return {
        "tokenization": tokenization_metrics,
        "padded": padded_metrics,
        "padding-free": padding_free_metrics
    }
    
if __name__ == "__main__":
    benchmark_datasets(
        dataset_path="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
        tokenizer_name="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        sequence_length=2048,
        batch_size=8,
        num_samples=16,
    )