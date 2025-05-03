import pyarrow.parquet as pq
from torch.utils.data import Dataset

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