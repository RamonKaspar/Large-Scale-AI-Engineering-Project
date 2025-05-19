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
    Adapted to be usable with DDP.
    """
    def __init__(self, parquet_file: str, sequence_length: int, bos_token_id: int = 1):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.token_buffer = []
        
        # Initialize rank and world_size to defaults
        self.rank = 0
        self.world_size = 1

    def __iter__(self):
        # Reset state
        self.token_buffer = []
        
        # Get worker info for single-process dataloader sharding
        worker_info = torch.utils.data.get_worker_info()
        
        # Get distributed info for DDP sharding
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
        # Further shard data if using multiple DataLoader workers
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            self.rank = self.rank * num_workers + worker_id
            self.world_size *= num_workers
            
        # Calculate the shard size and starting position for this rank
        shard_size = self.real_length // self.world_size
        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size if self.rank < self.world_size - 1 else self.real_length
        
        # Initialize to our shard's starting position
        self.current_index = start_idx
        # Track end of our shard
        self.end_index = end_idx
        
        return self

    def __next__(self):
        # Fill the buffer until we have enough tokens
        while len(self.token_buffer) < self.sequence_length + 1:
            if self.current_index >= self.end_index:
                # When reaching the end of this rank's shard, loop back to start of the shard
                self.current_index = self.rank * (self.real_length // self.world_size)
            
            # Get tokens from our shard
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