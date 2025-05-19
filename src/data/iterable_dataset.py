import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

class IterableParquetDataset(IterableDataset):
    """
    An iterable dataset that reads from a Parquet file and yields tokenized samples.
    Also works with DDP.
    """
    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        sequence_length: int,
        bos_token_id: int = 1
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.token_buffer = []

        # Initialize rank and world_size to defaults (will be updated in __iter__)
        self.rank = 0
        self.world_size = 1
        
    def __iter__(self):
        # Reset buffer and index when starting a new iteration
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
        
        # Initialize current_index to our shard's starting position
        self.current_index = start_idx
        # Track end of our shard
        self.end_index = end_idx
        
        # print(f"Rank {self.rank}/{self.world_size} processing documents {start_idx} to {end_idx-1}")
        
        return self
        
    def __next__(self):
        # Keep filling a buffer until we have enough tokens for a new sample.
        # Mask the loss for each token following the BoS token using -100 index.
        
        # Fill the buffer until we have enough tokens
        while len(self.token_buffer) < self.sequence_length + 1:
            if self.current_index >= self.end_index:
                # When reaching the end of this rank's shard, loop back to start of the shard
                self.current_index = self.rank * (self.real_length // self.world_size)
            
            # Get sample from our shard
            sample_str = str(self.parquet_ds["text"][self.current_index])
            self.current_index += 1
            
            # Add BOS token at document boundaries
            if not self.token_buffer or self.token_buffer[-1] != self.bos_token_id:
                self.token_buffer.append(self.bos_token_id)
                
            # Tokenize and add to buffer
            tokens = self.tokenizer.encode(sample_str)
            self.token_buffer.extend(tokens)
        
        # Prepare sequence-length sample from buffer
        token_sample = self.token_buffer[:self.sequence_length + 1]
        
        # Remove used tokens from the buffer
        if len(self.token_buffer) > self.sequence_length + 1:
            self.token_buffer = self.token_buffer[self.sequence_length:]
        else:
            self.token_buffer = []
        
        # Prepare inputs and labels
        inputs = torch.tensor(token_sample[:-1])
        labels = torch.tensor(token_sample[1:])
        
        # Mask the loss for the token after a BOS token
        # Find positions of BOS tokens in the inputs
        bos_positions = (inputs == self.bos_token_id).nonzero(as_tuple=True)[0]
        
        # For each BOS token position, mask the corresponding position in labels
        for pos in bos_positions:
            if pos < len(labels):
                labels[pos] = -100
        
        return inputs, labels