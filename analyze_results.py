"""
Analyze and visualize training results from log files (locally)
Used after running experiments on the cluster to generate performance plots.
"""

=== Analysis Feature 1: Pretokenization ===

import os
from analysis.plotting import plot_results

# Define log file paths
log_files = {
    'Baseline padded (batch_size=1)': 'logs/lsai_baseline_padded-396307.out',
    'Baseline padded (batch_size=2)': 'logs/lsai_baseline_padded-400129.out',
    'Baseline padding-free (batch_size=1)': 'logs/lsai_baseline_padding_free-396382.out',
    'Baseline padding-free (batch_size=2)': 'logs/lsai_baseline_padding_free-400137.out',
    'Pretokenized padded (batch_size=1)': 'logs/lsai_pretokenized_padded-399928.out',
    'Pretokenized padded (batch_size=2)': 'logs/lsai_pretokenized_padded-400185.out',
    'Pretokenized token-list (batch_size=1)': 'logs/lsai_pretokenized_token_list-399941.out',
    'Pretokenized token-list (batch_size=2)': 'logs/lsai_pretokenized_token_list-400186.out',
}
output_dir = "plots/plots_pretokenization"
os.makedirs(output_dir, exist_ok=True)

# Generate plots
plot_results(log_files, output_dir=output_dir)



# === Verification Feature 2: Distributed Data Parallel ===
from analysis.plotting import process_log_files, plot_time_series

log_files = {
    'Baseline padded (batch_size=1)': 'logs/lsai_baseline_padded-396307.out',
    'DDP Baseline padded (batch_size=1, world_size=1)': 'logs/ddp_verification/lsai_ddp_baseline_padded-447066.out',
    'Baseline padding-free (batch_size=1)': 'logs/lsai_baseline_padding_free-396382.out',
    'DDP Baseline padding-free (batch_size=1, world_size=1)': 'logs/ddp_verification/lsai_ddp_baseline_padding_free-447091.out',
    
}
output_dir = "plots_ddp/ddp_verification"
results, data_count = process_log_files(log_files)

print(f"\nSummary of Results (averaged over all steps except first 5):")
print("-" * 80)
print(f"{'Configuration':<55} {'Tokens/sec':<15} {'MFU %':<15} {'TFLOPs':<15}")
print("-" * 80)
for config, metrics in results.items():
    print(f"{config:<55} {metrics['tokens_per_second']:<15.2f} {metrics['mfu_percent']:<15.2f} {metrics['tflops']:<15.2f}")
    
plot_time_series(results, 'loss', 'Validation Loss During Training', 'Loss', 'loss_over_time.png', output_dir)
