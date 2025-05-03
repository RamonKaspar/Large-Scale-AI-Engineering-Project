import pyarrow.parquet as pq
from torch.utils.data import Dataset
import torch

class PreTokenizedDataset(Dataset):
    """
    Dataset for pretokenized data with padding.
    Uses the 'padded' format output from pretokenization.
    """
    def __init__(self, parquet_file: str, sequence_length: int, training_samples: int):
        # Load the pretokenized parquet file
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length
        self.training_samples = training_samples
        
        # Verify this is a pretokenized file in padded format
        if b'format' not in self.parquet_ds.schema.metadata:
            raise ValueError(f"The parquet file {parquet_file} does not appear to be pretokenized")
        
        format_type = self.parquet_ds.schema.metadata[b'format'].decode()
        if format_type != 'padded':
            raise ValueError(f"Expected 'padded' format, but got '{format_type}'")
        
        # Check for required columns
        required_columns = ["input_ids", "attention_mask"]
        missing_columns = [col for col in required_columns if col not in self.parquet_ds.column_names]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Loaded pretokenized dataset with {self.real_length} samples")

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        # Get the tokenized item directly from the parquet file
        actual_idx = idx % self.real_length
        
        # Extract input_ids and attention_mask
        input_ids = self.parquet_ds["input_ids"][actual_idx].as_py()
        attention_mask = self.parquet_ds["attention_mask"][actual_idx].as_py()
        
        # Return as dict to match the format expected by the existing collator
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        




class PreTokenizedTokenListDataset(Dataset):
    """
    Dataset for pretokenized token-list format.
    Uses the 'token-list' format output from pretokenization.
    """
    def __init__(self, parquet_file: str, sequence_length: int, training_samples: int):
        # Load the pretokenized parquet file
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length
        self.training_samples = training_samples
        
        # Verify this is a pretokenized file in token-list format
        if b'format' not in self.parquet_ds.schema.metadata:
            raise ValueError(f"The parquet file {parquet_file} does not appear to be pretokenized")
        
        format_type = self.parquet_ds.schema.metadata[b'format'].decode()
        if format_type != 'token-list':
            raise ValueError(f"Expected 'token-list' format, but got '{format_type}'")
        
        # Check for required columns
        if "tokens" not in self.parquet_ds.column_names:
            raise ValueError("Missing required column: tokens")
        
        print(f"Loaded token-list dataset with {self.real_length} samples")

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        # Get the tokenized item directly from the parquet file
        actual_idx = idx % self.real_length
        
        # Extract tokens
        tokens = self.parquet_ds["tokens"][actual_idx].as_py()
        
        # Return in a format compatible with the collator
        return tokens



class TokenListCollator:
    """
    Collator for token-list format that handles variable length sequences
    and creates proper input_ids and labels.
    """
    def __init__(self, sequence_length: int, bos_token_id: int = 1):
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        
    def __call__(self, examples):
        batch_size = len(examples)
        
        # Create input tensors and labels
        inputs = torch.ones((batch_size, self.sequence_length), dtype=torch.long)
        labels = torch.ones((batch_size, self.sequence_length), dtype=torch.long)
        
        for i, tokens in enumerate(examples):
            # If we have enough tokens
            if len(tokens) > 1:  # Need at least 2 tokens (1 for input, 1 for label)
                # Limit to sequence_length + 1 (for both input and target)
                tokens = tokens[:self.sequence_length + 1]
                
                # Split into inputs and labels
                seq_len = min(len(tokens) - 1, self.sequence_length)
                inputs[i, :seq_len] = torch.tensor(tokens[:seq_len])
                labels[i, :seq_len] = torch.tensor(tokens[1:seq_len+1])
                
                # Find BOS positions to mask in labels
                bos_positions = [j for j, t in enumerate(tokens[:seq_len]) if t == self.bos_token_id]
                for pos in bos_positions:
                    if pos < seq_len:
                        labels[i, pos] = -100
                
                # Mask remaining positions
                if seq_len < self.sequence_length:
                    labels[i, seq_len:] = -100
        
        return inputs, labels




class PreTokenizedPackedDataset(Dataset):
    """
    Dataset for pretokenized packed format.
    Uses the 'packed' format output from pretokenization.
    """
    def __init__(self, parquet_file: str, sequence_length: int, training_samples: int, bos_token_id: int = 1):
        # Load the pretokenized parquet file
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        
        # Calculate actual training samples based on total tokens
        total_chunks = len(self.parquet_ds)
        if total_chunks == 0:
            raise ValueError("Empty packed dataset")
            
        # Verify this is a pretokenized file in packed format
        if b'format' not in self.parquet_ds.schema.metadata:
            raise ValueError(f"The parquet file {parquet_file} does not appear to be pretokenized")
        
        format_type = self.parquet_ds.schema.metadata[b'format'].decode()
        if format_type != 'packed':
            raise ValueError(f"Expected 'packed' format, but got '{format_type}'")
        
        # Get chunk size from metadata
        if b'chunk_size' in self.parquet_ds.schema.metadata:
            self.chunk_size = int(self.parquet_ds.schema.metadata[b'chunk_size'].decode())
        else:
            # Infer from first chunk
            first_chunk = self.parquet_ds["packed_chunks"][0].as_py()
            self.chunk_size = len(first_chunk)
            
        # Calculate how many sequence-sized samples we can extract
        self.samples_per_chunk = self.chunk_size // self.sequence_length
        self.total_samples = total_chunks * self.samples_per_chunk
        self.training_samples = min(training_samples, self.total_samples)
        
        print(f"Loaded packed dataset with {total_chunks} chunks, {self.samples_per_chunk} samples per chunk, "
              f"{self.total_samples} total samples")

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        # Convert sample index to chunk index and position within chunk
        actual_idx = idx % self.total_samples
        chunk_idx = actual_idx // self.samples_per_chunk
        sample_idx_in_chunk = actual_idx % self.samples_per_chunk
        
        # Calculate start and end positions for this sample
        start_pos = sample_idx_in_chunk * self.sequence_length
        end_pos = start_pos + self.sequence_length + 1  # +1 for next token prediction
        
        # Get the chunk
        chunk = self.parquet_ds["packed_chunks"][chunk_idx].as_py()
        bos_positions = self.parquet_ds["bos_positions"][chunk_idx].as_py()
        
        # Extract the sequence
        if end_pos <= len(chunk):
            # Get tokens for this sequence
            token_sequence = chunk[start_pos:end_pos]
            
            # Find BOS positions within this sequence
            # First adjust positions to be relative to this sequence
            seq_bos_positions = [pos - start_pos for pos in bos_positions 
                               if start_pos <= pos < end_pos]
            
            # Create input_ids and labels
            inputs = token_sequence[:-1]
            labels = token_sequence[1:]
            
            # Mask labels after BOS tokens (we don't want to predict the first token after BOS)
            for pos in seq_bos_positions:
                if 0 <= pos < len(labels):
                    labels[pos] = -100
                    
            return torch.tensor(inputs), torch.tensor(labels)
        else:
            # Handle edge case - if the sequence extends beyond the chunk
            # This should be rare with properly sized chunks
            remaining = len(chunk) - start_pos
            token_sequence = chunk[start_pos:]
            
            # Pad to required length if needed
            if remaining < self.sequence_length + 1:
                pad_token_id = 0  # Use 0 as default pad
                token_sequence += [pad_token_id] * (self.sequence_length + 1 - remaining)
            
            # Find BOS positions for this sequence
            seq_bos_positions = [pos - start_pos for pos in bos_positions 
                               if start_pos <= pos < len(chunk)]
            
            # Create input_ids and labels
            inputs = token_sequence[:-1]
            labels = token_sequence[1:]
            
            # Mask labels after BOS tokens
            for pos in seq_bos_positions:
                if 0 <= pos < len(labels):
                    labels[pos] = -100
                    
            # Also mask padding in labels (if any)
            for i in range(remaining - 1, len(labels)):
                labels[i] = -100
                
            return torch.tensor(inputs), torch.tensor(labels)