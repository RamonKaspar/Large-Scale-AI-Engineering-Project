import os

# === Compare baseline implementations ===
# Compare padding vs padding-free baseline implementations

from analysis.benchmark_datasets import benchmark_datasets

# Run on login node for quick benchmarking
benchmark_datasets(
    dataset_path="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
    tokenizer_name="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    sequence_length=2048,
    batch_size=8,
    num_samples=16,
)



# === Plot results ===
from analysis.plotting import plot_results

# Define log file paths
log_files = {
    'Baseline padded': 'logs/lsai_baseline_padded-396307.out',
    'Baseline padding-free': 'logs/lsai_baseline_padding_free-396382.out',
    'Pretokenized padded': 'logs/lsai_pretokenized_padded-397348.out',
}
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plot_results(log_files, output_dir=output_dir)