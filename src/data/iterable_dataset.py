import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

class IterableParquetDataset(IterableDataset):
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
        self.current_index = 0
        self.token_buffer = []

    def __iter__(self):
        # Reset buffer and index when starting a new iteration
        self.token_buffer = []
        self.current_index = 0
        return self

    def __next__(self):
        # Keep filling a buffer until we have enough tokens for a new sample.
        # Mask the loss for each token following the BoS token using -100 index.
        
        # Fill the buffer until we have enough tokens
        while len(self.token_buffer) < self.sequence_length + 1:
            # Get the next document
            if self.current_index >= self.real_length:
                self.current_index = 0  # Wrap around to the beginning
            
            # Get text from current document
            sample_str = str(self.parquet_ds["text"][self.current_index])
            self.current_index += 1
            
            # Tokenize the document
            tokens = self.tokenizer.encode(sample_str)
            
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
        
        # Prepare inputs and labels
        inputs = torch.tensor(token_sample[:-1])
        labels = torch.tensor(token_sample[1:])
        
        # Mask the loss for the token after a BOS token
        # Find positions of BOS tokens in the inputs
        bos_positions = (inputs == self.bos_token_id).nonzero(as_tuple=True)[0]
        
        # For each BOS token position, mask the corresponding position in labels
        for pos in bos_positions:
            if pos + 1 < len(labels):
                labels[pos] = -100
        
        return inputs, labels