from analysis.benchmark_datasets import benchmark_datasets

# Compare padding vs padding-free baseline implementations
# Run on login node for quick benchmarking
benchmark_datasets(
    dataset_path="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
    tokenizer_name="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    sequence_length=2048,
    batch_size=8,
    num_samples=16,
)

