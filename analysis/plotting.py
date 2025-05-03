import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="darkgrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
PALETTE = sns.color_palette()


def extract_metrics(log_file):
    """
    Extract performance metrics from a training log file.
    
    Args:
        log_file (str): Path to the log file.
        
    Returns:
        dict: A dictionary containing the extracted metrics.
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Regular expression pattern to extract metrics
    pattern = r"Step: (\d+) \| Loss: (.+?) \| Tokens per second: (.+?) \| Training tokens per second \(%\): (.+?) \| MFU \(%\): (.+?) \| TFLOPs: (.+?)$"
    
    # Extract all metrics
    matches = re.findall(pattern, content, re.MULTILINE)
    if not matches:
        print(f"No metrics found in {log_file}")
        return None
    
    # Convert to numeric values
    steps = [int(m[0]) for m in matches]
    loss = [float(m[1]) for m in matches]
    tokens_per_second = [float(m[2]) for m in matches]
    training_tokens_percent = [float(m[3]) for m in matches]
    mfu_percent = [float(m[4]) for m in matches]
    tflops = [float(m[5]) for m in matches]
    
    # Skip the first few steps as they often have outlier performance due to initialization overhead
    print(f"INFO: Skipping the first {min(5, len(steps))} steps for averaging (warm-up)")
    skip_steps = min(5, len(steps))
    
    # Calculate averages (excluding the first few steps)
    avg_tokens_per_second = np.mean(tokens_per_second[skip_steps:])
    avg_mfu_percent = np.mean(mfu_percent[skip_steps:])
    avg_tflops = np.mean(tflops[skip_steps:])
    
    return {
        'steps': steps,
        'loss': loss,
        'all_tokens_per_second': tokens_per_second,
        'all_training_tokens_percent': training_tokens_percent,
        'all_mfu_percent': mfu_percent,
        'all_tflops': tflops,
        'tokens_per_second': avg_tokens_per_second,
        'mfu_percent': avg_mfu_percent,
        'tflops': avg_tflops
    }



def process_log_files(log_files):
    """
    Process multiple log files and extract metrics.
    
    Args:
        log_files (dict): Dictionary of configuration names and log file paths.
        
    Returns:
        dict: Dictionary of configuration names and their extracted metrics.
        int: Count of data points processed.
    """
    results = {}
    data_count = 0
    
    for config, log_file in log_files.items():
        metrics = extract_metrics(log_file)
        if metrics:
            results[config] = metrics
            data_count += len(metrics['steps'])
            print(f"Processed {config}: {len(metrics['steps'])} data points found")
    
    return results, data_count



def plot_bar_comparison(results, metric_name, title, ylabel, filename, output_dir="plots"):
    """
    Create a bar chart comparing a metric across configurations.
    """
    plt.figure(figsize=(10, 6))
    
    configs = list(results.keys())
    values = [results[c][metric_name] for c in configs]
    
    df = pd.DataFrame({
        'Configuration': configs,
        metric_name: values
    })
    
    ax = sns.barplot(x='Configuration', y=metric_name, hue='Configuration', data=df, palette=PALETTE[:len(configs)], legend=False)
    plt.title(title, fontsize=18, pad=20)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=8)
    ax.set_xticklabels([config.replace(' ', '\n').replace('_', ' ') for config in configs])
    plt.yticks(fontsize=12)
    plt.xlabel(None)  # Remove x-label
    
    # Add values on top of bars
    for i, v in enumerate(values):
        ax.text(i, v * 1.01, f"{v:.2f}", ha='center', fontsize=13, fontweight='bold')
    
    # Extend the upper limit to make room for labels
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)  # Add 10% padding at the top
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()



def plot_time_series(results, metric_name, title, ylabel, filename, output_dir="plots"):
    """
    Create a line chart showing metrics over time.
    
    Args:
        results (dict): Dictionary of configuration names and their metrics.
        metric_name (str): Name of the metric to plot.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        filename (str): Output filename.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(16, 6))
    
    # Create the line plot
    for i, (config, metrics) in enumerate(results.items()):
        plt.plot(metrics['steps'], metrics[metric_name], label=config, 
                 color=PALETTE[i % len(PALETTE)], marker='o', markersize=4, markevery=5)
    
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()



def plot_efficiency_comparison(results, output_dir="plots"):
    """Create an improved comparison chart for efficiency metrics with consistent coloring."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    
    configs = list(results.keys())
    tokens_per_second = [results[c]['tokens_per_second'] for c in configs]
    mfu_percent = [results[c]['mfu_percent'] for c in configs]
    
    df_tokens = pd.DataFrame({
        'Configuration': configs,
        'Tokens/sec': tokens_per_second
    })
    df_mfu = pd.DataFrame({
        'Configuration': configs,
        'MFU %': mfu_percent
    })
    
    # Plot 1: Tokens per second
    sns.barplot(x='Configuration', y='Tokens/sec', hue='Configuration', 
               data=df_tokens, palette=PALETTE[:len(configs)], ax=ax1, legend=False)
    ax1.set_title('Throughput (Tokens/second)', fontsize=16)
    ax1.set_ylabel('Tokens per Second', fontsize=14)
    ax1.set_xlabel(None)  # Remove x-label from first subplot
    ax1.set_xticklabels([config.replace(' ', '\n').replace('_', ' ') for config in configs])
    # Add value labels
    for i, v in enumerate(tokens_per_second):
        ax1.text(i, v * 1.01, f'{v:.1f}', ha='center', fontsize=13, fontweight='bold')
    
    # Plot 2: MFU percentage
    sns.barplot(x='Configuration', y='MFU %', hue='Configuration', 
               data=df_mfu, palette=PALETTE[:len(configs)], ax=ax2, legend=False)
    ax2.set_title('GPU Utilization (MFU %)', fontsize=16)
    ax2.set_ylabel('Model FLOPS Utilization (%)', fontsize=14)
    ax2.set_xlabel(None)  # Remove x-label from second subplot
    ax2.set_xticklabels([config.replace(' ', '\n').replace('_', ' ') for config in configs])
    # Add value labels
    for i, v in enumerate(mfu_percent):
        ax2.text(i, v * 1.01, f'{v:.1f}%', ha='center', fontsize=13, fontweight='bold')
    
    # Extend the upper limit to make room for labels
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min, y_max * 1.1)  # Add 10% padding at the top
    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min, y_max * 1.1)  # Add 10% padding at the top
    
    fig.suptitle('Performance Comparison: Throughput and Efficiency', fontsize=18, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    
def plot_batch_comparison(results, output_dir="plots"):
    """Create grouped bar charts comparing batch sizes within each approach."""
    
    # Define the approaches and their configurations
    approaches = {
        "Baseline Padded": ["Baseline padded (batch_size=1)", "Baseline padded (batch_size=2)"],
        "Baseline Padding-Free": ["Baseline padding-free (batch_size=1)", "Baseline padding-free (batch_size=2)"],
        "Pretokenized Padded": ["Pretokenized padded (batch_size=1)", "Pretokenized padded (batch_size=2)"],
        "Pretokenized Token-List": ["Pretokenized token-list (batch_size=1)", "Pretokenized token-list (batch_size=2)"]
    }
    
    # Create figure with subplots for tokens/sec and MFU
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Define x positions for the groups
    x = np.arange(len(approaches))
    width = 0.35  # Width of the bars
    
    # Extract data
    batch1_tokens = []
    batch2_tokens = []
    batch1_mfu = []
    batch2_mfu = []
    
    for approach, configs in approaches.items():
        if configs[0] in results:
            batch1_tokens.append(results[configs[0]]['tokens_per_second'])
            batch1_mfu.append(results[configs[0]]['mfu_percent'])
        else:
            batch1_tokens.append(0)
            batch1_mfu.append(0)
            
        if configs[1] in results:
            batch2_tokens.append(results[configs[1]]['tokens_per_second'])
            batch2_mfu.append(results[configs[1]]['mfu_percent'])
        else:
            batch2_tokens.append(0)
            batch2_mfu.append(0)
    
    # Plot tokens per second
    ax1.bar(x - width/2, batch1_tokens, width, label='Batch Size = 1', color=PALETTE[-1])
    ax1.bar(x + width/2, batch2_tokens, width, label='Batch Size = 2', color=PALETTE[-2])
    
    # Plot MFU
    ax2.bar(x - width/2, batch1_mfu, width, label='Batch Size = 1', color=PALETTE[-1])
    ax2.bar(x + width/2, batch2_mfu, width, label='Batch Size = 2', color=PALETTE[-2])
    
    # Add value labels
    for i, v in enumerate(batch1_tokens):
        ax1.text(i - width/2, v * 1.01, f'{v:.1f}', ha='center', fontsize=12)
    for i, v in enumerate(batch2_tokens):
        ax1.text(i + width/2, v * 1.01, f'{v:.1f}', ha='center', fontsize=12)
        
    for i, v in enumerate(batch1_mfu):
        ax2.text(i - width/2, v * 1.01, f'{v:.1f}%', ha='center', fontsize=12)
    for i, v in enumerate(batch2_mfu):
        ax2.text(i + width/2, v * 1.01, f'{v:.1f}%', ha='center', fontsize=12)
    
    # Configure plot 1 (tokens per second)
    ax1.set_ylabel('Tokens per Second', fontsize=14)
    ax1.set_title('Throughput Comparison: Batch Size 1 vs. Batch Size 2', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels([])  # Hide x-labels for top plot
    ax1.legend(fontsize=12)
    
    # Configure plot 2 (MFU)
    ax2.set_ylabel('Model FLOPS Utilization (%)', fontsize=14)
    ax2.set_title('Efficiency Comparison: Batch Size 1 vs. Batch Size 2', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(approaches.keys(), fontsize=12)
    ax2.legend(fontsize=12)
    
    # Extend the upper limit to make room for labels
    for ax in [ax1, ax2]:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.1)
    
    fig.suptitle('Performance Impact of Batch Size by Implementation Approach', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_size_comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()



def plot_results(log_files, output_dir="plots"):
    """
    Generate and save visualization plots from training logs.
    
    Args:
        log_files (dict): Dictionary of configuration names and log file paths.
        output_dir (str): Directory to save the plots.
    """
    results, data_count = process_log_files(log_files)
    if not results:
        print("No data found in log files. Exiting.")
        return
    
    print(f"\nSummary of Results (averaged over all steps except first 5):")
    print("-" * 80)
    print(f"{'Configuration':<35} {'Tokens/sec':<15} {'MFU %':<15} {'TFLOPs':<15}")
    print("-" * 80)
    
    for config, metrics in results.items():
        print(f"{config:<35} {metrics['tokens_per_second']:<15.2f} {metrics['mfu_percent']:<15.2f} {metrics['tflops']:<15.2f}")
    
    # === Create plots ===
    # 1. Bar chart of tokens per second
    plot_bar_comparison(
        results, 
        'tokens_per_second', 
        'Average Tokens per Second by Configuration',
        'Tokens per Second',
        'tokens_per_second_comparison.png',
        output_dir
    )
    
    # 2. Bar chart of MFU percentage
    plot_bar_comparison(
        results, 
        'mfu_percent', 
        'Model FLOPS Utilization (MFU) by Configuration',
        'MFU (%)',
        'mfu_comparison.png',
        output_dir
    )
    
    # 3. Bar chart of TFLOPs
    plot_bar_comparison(
        results, 
        'tflops', 
        'Average TFLOPs by Configuration',
        'TFLOPs',
        'tflops_comparison.png',
        output_dir
    )
    
    # 4. Line chart of tokens per second over time
    plot_time_series(
        results, 
        'all_tokens_per_second', 
        'Tokens per Second During Training',
        'Tokens per Second',
        'tokens_per_second_over_time.png',
        output_dir
    )
    
    # 5. Line chart of MFU over time
    plot_time_series(
        results, 
        'all_mfu_percent', 
        'Model FLOPS Utilization During Training',
        'MFU (%)',
        'mfu_over_time.png',
        output_dir
    )
    
    # 6. Special comparison chart
    plot_efficiency_comparison(results, output_dir)
    
    # 6. Line chart of loss over time
    plot_time_series(
        results,
        'loss',
        'Validation Loss During Training',
        'Loss',
        'loss_over_time.png',
        output_dir
    )
    
    # 7. Batch size comparison
    plot_batch_comparison(results, output_dir)
