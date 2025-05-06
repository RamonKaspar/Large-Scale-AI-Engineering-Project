import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset
import torch

class PreTokenizedDataset(Dataset):
    """
    Minimal adaptation of ParquetDataset that uses pretokenized data.
    """
    def __init__(self, parquet_file: str, sequence_length: int, training_samples: int):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length
        self.training_samples = training_samples
        print(f"Loaded pretokenized dataset with {self.real_length} samples")

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        actual_idx = idx % self.real_length
        # Simply return the already-tokenized data
        return {
            "input_ids": self.parquet_ds["input_ids"][actual_idx].as_py(),
            "attention_mask": self.parquet_ds["attention_mask"][actual_idx].as_py()
        }
        


class IterablePreTokenizedDataset(IterableDataset):
    """
    Minimal adaptation of IterableParquetDataset that uses pretokenized data.
    """
    def __init__(self, parquet_file: str, sequence_length: int, bos_token_id: int = 1):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.current_index = 0
        self.token_buffer = []

    def __iter__(self):
        self.token_buffer = []
        self.current_index = 0
        return self

    def __next__(self):
        # Fill the buffer until we have enough tokens
        while len(self.token_buffer) < self.sequence_length + 1:
            # Get the next document
            if self.current_index >= self.real_length:
                self.current_index = 0  # Wrap around to the beginning
            
            tokens = self.parquet_ds["tokens"][self.current_index].as_py()
            self.current_index += 1
            
            # Add BOS token at the beginning of the document
            if not self.token_buffer or self.token_buffer[-1] != self.bos_token_id:
                self.token_buffer.append(self.bos_token_id)
                
            # Add tokens to the buffer
            self.token_buffer.extend(tokens)
        
        # Take sequence_length + 1 tokens from the buffer
        token_sample = self.token_buffer[:self.sequence_length + 1]
        
        # Remove used tokens from the buffer
        if len(self.token_buffer) > self.sequence_length + 1:
            self.token_buffer = self.token_buffer[self.sequence_length:]
        else:
            self.token_buffer = []
        
        inputs = torch.tensor(token_sample[:-1])
        labels = torch.tensor(token_sample[1:])
        
        # Mask the loss for the token after a BOS token
        bos_positions = (inputs == self.bos_token_id).nonzero(as_tuple=True)[0]
        for pos in bos_positions:
            if pos < len(labels):
                labels[pos] = -100
        
        return inputs, labels