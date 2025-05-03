"""
Analyze and visualize training results from log files.
Used after running experiments on the cluster to generate performance plots.
"""

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
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Generate plots
plot_results(log_files, output_dir=output_dir)
