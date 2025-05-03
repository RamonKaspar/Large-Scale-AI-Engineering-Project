"""
Pretokenization Verification Script (run this on login node)
This script verifies that the two pretokenized (padded and token-list) formats match on-the-fly tokenization.
"""

import random
import pyarrow.parquet as pq
from transformers import AutoTokenizer


TOKENIZER_NAME = "unsloth/Mistral-Nemo-Base-2407-bnb-4bit"
ORIGINAL_DATA_PATH = "/capstor/scratch/cscs/kasparr/project/train_data.parquet"
PRETOKENIZED_PADDED_PATH = "/capstor/scratch/cscs/kasparr/project/train_data_tokenized_padded_snappy.parquet"
PRETOKENIZED_TOKEN_LIST_PATH = "/capstor/scratch/cscs/kasparr/project/train_data_tokenized_token-list_snappy.parquet"
MODEL_SEQUENCE_LENGTH = 2048


def load_tokenizer(tokenizer_name: str):
    """Load the tokenizer and return it."""
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


def load_original_data(original_path: str):
    """Load the original text data."""
    print(f"Loading original text data: {original_path}")
    try:
        original_table = pq.read_table(original_path)
        num_samples = len(original_table)
        print(f"Found {num_samples} original documents.")
        return original_table, num_samples
    except FileNotFoundError:
        print(f"Error: Original data file not found at {original_path}")
        return None, 0
    except Exception as e:
        print(f"Error loading original data: {e}")
        return None, 0


def load_pretokenized_data(pretokenized_path: str, expected_format: str):
    """Load pretokenized data."""
    print(f"Loading pretokenized {expected_format} data: {pretokenized_path}")
    try:
        pretok_table = pq.read_table(pretokenized_path)
        
        # Check format type from metadata
        if b'format' in pretok_table.schema.metadata:
            format_type = pretok_table.schema.metadata[b'format'].decode()
            if format_type != expected_format:
                print(f"Warning: Expected format '{expected_format}', but found '{format_type}'")
        
        # Check required columns based on format type
        if expected_format == "padded":
            required_cols = ["input_ids", "attention_mask"]
        elif expected_format == "token-list":
            required_cols = ["tokens"]
        elif expected_format == "packed":
            required_cols = ["packed_chunks", "bos_positions"]
        else:
            required_cols = []
            
        missing_cols = [col for col in required_cols if col not in pretok_table.column_names]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None
            
        return pretok_table
    except FileNotFoundError:
        print(f"Error: Pretokenized data file not found at {pretokenized_path}")
        return None
    except Exception as e:
        print(f"Error loading pretokenized data: {e}")
        return None


def verify_padded_format(sample_idx: int = -1):
    """Verify the padded format."""
    print("\n" + "="*80)
    print("VERIFYING PADDED FORMAT")
    print("="*80 + "\n")
    
    # 1. Load resources
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    if not tokenizer:
        return False
        
    original_table, num_samples = load_original_data(ORIGINAL_DATA_PATH)
    if not original_table:
        return False
        
    pretok_table = load_pretokenized_data(PRETOKENIZED_PADDED_PATH, "padded")
    if not pretok_table:
        return False
    
    # 2. Select sample index
    if sample_idx == -1:
        sample_idx = random.randint(0, min(len(original_table), len(pretok_table)) - 1)
        print(f"Selected random sample index: {sample_idx}")
    else:
        print(f"Using specified sample index: {sample_idx}")

    # 3. Get original text
    original_text = original_table["text"][sample_idx].as_py()
    print("\n--- Original Text Sample ---")
    print(original_text[:200] + "..." if len(original_text) > 200 else original_text)

    # 4. Perform on-the-fly tokenization 
    padded_length = MODEL_SEQUENCE_LENGTH + 1  # +1 for collator compatibility
    print(f"\n--- On-the-Fly Tokenization (length={padded_length}) ---")
    on_the_fly_encoded = tokenizer.encode_plus(
        original_text,
        max_length=padded_length,
        padding='max_length',
        truncation=True,
        return_tensors=None
    )
    on_the_fly_ids = on_the_fly_encoded["input_ids"]
    on_the_fly_mask = on_the_fly_encoded["attention_mask"]
    
    print(f"Input IDs (first 10): {on_the_fly_ids[:10]}")
    print(f"Input IDs (last 10): {on_the_fly_ids[-10:]}")
    
    # 5. Get pretokenized data
    print("\n--- Pretokenized Data ---")
    pretok_ids = pretok_table["input_ids"][sample_idx].as_py()
    pretok_mask = pretok_table["attention_mask"][sample_idx].as_py()
    
    print(f"Input IDs (first 10): {pretok_ids[:10]}")
    print(f"Input IDs (last 10): {pretok_ids[-10:]}")
    
    # 6. Compare
    print("\n--- Comparison ---")
    ids_match = (on_the_fly_ids == pretok_ids)
    mask_match = (on_the_fly_mask == pretok_mask)
    
    if ids_match and mask_match:
        return True
    else:
        return False


def verify_token_list_format(sample_idx: int = -1):
    """Verify the token-list format."""
    print("\n" + "="*80)
    print("VERIFYING TOKEN-LIST FORMAT")
    print("="*80 + "\n")
    
    # 1. Load resources
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    if not tokenizer:
        return False
        
    original_table, num_samples = load_original_data(ORIGINAL_DATA_PATH)
    if not original_table:
        return False
        
    pretok_table = load_pretokenized_data(PRETOKENIZED_TOKEN_LIST_PATH, "token-list")
    if not pretok_table:
        return False
    
    # 2. Select sample index
    if sample_idx == -1:
        sample_idx = random.randint(0, min(len(original_table), len(pretok_table)) - 1)
        print(f"Selected random sample index: {sample_idx}")
    else:
        print(f"Using specified sample index: {sample_idx}")

    # 3. Get original text
    original_text = original_table["text"][sample_idx].as_py()
    print("\n--- Original Text Sample ---")
    print(original_text[:200] + "..." if len(original_text) > 200 else original_text)

    # 4. Perform on-the-fly tokenization without padding
    print(f"\n--- On-the-Fly Tokenization (no padding) ---")
    on_the_fly_tokens = tokenizer.encode(
        original_text,
        max_length=MODEL_SEQUENCE_LENGTH,
        truncation=True
    )
    
    print(f"Token IDs (first 10): {on_the_fly_tokens[:10]}")
    print(f"Token IDs (last 10): {on_the_fly_tokens[-10:]}")
    print(f"Total tokens: {len(on_the_fly_tokens)}")
    
    # 5. Get pretokenized data
    print("\n--- Pretokenized Data ---")
    pretok_tokens = pretok_table["tokens"][sample_idx].as_py()
    
    print(f"Token IDs (first 10): {pretok_tokens[:10]}")
    print(f"Token IDs (last 10): {pretok_tokens[-10:]}")
    print(f"Total tokens: {len(pretok_tokens)}")
    
    # 6. Compare
    print("\n--- Comparison ---")
    tokens_match = (on_the_fly_tokens == pretok_tokens)
    
    if tokens_match:
        return True
    else:
        return False


def main():
    padded_success = False
    token_list_success = False
    
    padded_success = verify_padded_format()
    if padded_success: print("Padded format verification passed.")
    
    token_list_success = verify_token_list_format()
    if token_list_success: print("Token-list format verification passed.")

if __name__ == "__main__":
    main()