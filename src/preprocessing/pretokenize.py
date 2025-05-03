"""
This module provides functionality to convert text data into tokenized padded format
for offline tokenization (for ParquetDataset).
"""

import os
import time
import argparse
from typing import Dict, List, Any
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pretokenize a parquet dataset of text")
    parser.add_argument(
        "--input-path",
        type=str,
        default="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
        help="Path to the input parquet file containing the 'text' column",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/capstor/store/cscs/ethz/large-sc/datasets",
        help="Directory to save the pretokenized output files",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        help="Name or path of the tokenizer to use",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(    # TODO: Maybe add padding-free in future
        "--format",
        type=str,
        nargs="+",
        choices=["padded"],
        default=["padded"],
        help="Which tokenization formats to generate",
    )
    return parser.parse_args()


def tokenize_text_padded(texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int, batch_size: int) -> Dict[str, List[Any]]:
    """
    Tokenize a list of texts with padding for use with ParquetDataset.
    
    Args:
        texts: List of text strings to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        
    Returns:
        Dictionary containing tokenized data with input_ids and attention_mask
    """
    total = len(texts)
    tokenized_data = {
        "input_ids": [],
        "attention_mask": []
    }
    
    start_time = time.time()
    
    for i in tqdm(range(0, total, batch_size), desc="Tokenizing texts (padded format)"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with padding
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # Return Python lists
        )
        
        tokenized_data["input_ids"].extend(encoded["input_ids"])
        tokenized_data["attention_mask"].extend(encoded["attention_mask"])

        # Optional progress info
        if (i // batch_size) % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            docs_per_sec = i / elapsed
            print(f"Processed {i}/{total} documents ({docs_per_sec:.2f} docs/sec)")
    
    elapsed = time.time() - start_time
    print(f"Tokenization completed in {elapsed:.2f} seconds ({total/elapsed:.2f} docs/sec)")

    return tokenized_data


def save_tokenized_data(data: Dict[str, Any], output_path: str, tokenizer_name: str):
    """
    Save tokenized data to a parquet file with appropriate metadata.
    
    Args:
        data: The tokenized data
        output_path: Path to save the output
        tokenizer_name: Name of the tokenizer used
    """
    print(f"Converting padded format to PyArrow table")
    
    # Create table for padded format
    table = pa.Table.from_pydict({
        "input_ids": data["input_ids"],
        "attention_mask": data["attention_mask"]
    })
    
    # Add metadata
    metadata = {
        "tokenizer": tokenizer_name,
        "format": "padded",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Encode metadata for PyArrow
    metadata_encoded = {k: v.encode() for k, v in metadata.items()}
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create table with metadata
    table = table.replace_schema_metadata(metadata_encoded)
    
    # Save the tokenized data
    print(f"Saving padded format to: {output_path}")
    # NOTE: High compression level to not exceed disk quota limitations
    pq.write_table(table, output_path, compression="zstd", compression_level=22)
    
    # Report output size
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Output size: {output_size_mb:.2f} MB")


def main():
    """Main function to pretokenize the dataset."""
    args = parse_args()
    
    # Load the tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Load the parquet file
    print(f"Loading parquet dataset from: {args.input_path}")
    parquet_table = pq.read_table(args.input_path)
    
    # Extract text column as list
    texts = parquet_table["text"].to_pylist()
    print(f"Loaded {len(texts)} text documents")
    
    # Measure data size and estimate tokenized size
    input_size_mb = sum(len(t.encode('utf-8')) for t in texts[:1000]) / 1000 / 1000
    estimated_full_input_size_gb = input_size_mb * len(texts) / 1000
    print(f"Estimated input text size: {estimated_full_input_size_gb:.2f} GB")
    
    # Process padded format only
    output_file = "train_data_tokenized_padded.parquet"
    output_path = os.path.join(args.output_dir, output_file)
    
    # Add +1 to max_length for collator compatibility (as in verification script)
    padded_length_for_tokenizer = args.max_length + 1
    data = tokenize_text_padded(
        texts, tokenizer, padded_length_for_tokenizer, args.batch_size
    )
    
    save_tokenized_data(data, output_path, args.tokenizer_name)
    
    print(f"Pretokenization completed successfully!")


if __name__ == "__main__":
    main()