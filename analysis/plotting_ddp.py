import re
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.plotting import plot_time_series, plot_bar_comparison


sns.set_theme(style="darkgrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
PALETTE = sns.color_palette()


def extract_ddp_metrics(log_file):
    """
    Extract performance metrics from a training log file, supporting both old and new DDP formats.
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract world_size from the log content
    world_size_match = re.search(r"Distributed training enabled: (\d+) processes", content)
    world_size = int(world_size_match.group(1)) if world_size_match else 1
    
    # Find global batch size
    global_batch_match = re.search(r"Global batch size: (\d+)", content)
    global_batch_size = int(global_batch_match.group(1)) if global_batch_match else world_size
    
    pattern = r"Step: (\d+) \| Loss: (.+?) \| Tokens per second: (.+?) \| Training tokens per second \(%\): (.+?) \| MFU \(%\): (.+?) \| TFLOPs: (.+?)(?:\s\|\sGlobal batch size:\s(\d+)\s\|\sGlobal tokens/sec:\s(.+?)(?:\s\|\sGlobal MFU \(%\):\s(.+?)\s\|\sGlobal TFLOPs:\s(.+?))?)?\s?\|"
    
    matches = re.findall(pattern, content, re.MULTILINE)
    if not matches:
        print(f"No metrics found in {log_file}")
        return None
    
    steps = [int(m[0]) for m in matches]
    loss = [float(m[1]) for m in matches]
    
    per_device_tokens_per_second = [float(m[2]) for m in matches]
    training_tokens_percent = [float(m[3]) for m in matches]
    per_device_mfu_percent = [float(m[4]) for m in matches]
    per_device_tflops = [float(m[5]) for m in matches]
    
    has_global_metrics = len(matches[0]) > 7 and matches[0][7] != ''
    
    if has_global_metrics:
        global_tokens_per_second = [float(m[7]) for m in matches]
        
        # Check if we have the newer format with Global MFU and Global TFLOPs
        if len(matches[0]) > 9 and matches[0][8] != '' and matches[0][9] != '':
            global_mfu_percent = [float(m[8]) for m in matches]
            global_tflops = [float(m[9]) for m in matches]
        else:
            global_mfu_percent = [per_device_mfu * world_size for per_device_mfu in per_device_mfu_percent]
            global_tflops = [per_device_tflop * world_size for per_device_tflop in per_device_tflops]
    else:
        global_tokens_per_second = None
        global_mfu_percent = None
        global_tflops = None
        
    # Skip the first 5 steps
    print(f"INFO: Skipping the first {min(5, len(steps))} steps for averaging (warm-up)")
    skip_steps = min(5, len(steps))
    
    # Calculate averages
    avg_per_device_tokens_per_second = np.mean(per_device_tokens_per_second[skip_steps:])
    avg_per_device_mfu_percent = np.mean(per_device_mfu_percent[skip_steps:])
    avg_per_device_tflops = np.mean(per_device_tflops[skip_steps:])
    
    avg_global_tokens_per_second = np.mean(global_tokens_per_second[skip_steps:])
    avg_global_mfu_percent = np.mean(global_mfu_percent[skip_steps:])
    avg_global_tflops = np.mean(global_tflops[skip_steps:])
    
    return {
        'steps': steps,
        'loss': loss,
        'world_size': world_size,
        'global_batch_size': global_batch_size,
        'all_per_device_tokens_per_second': per_device_tokens_per_second,
        'all_training_tokens_percent': training_tokens_percent,
        'all_per_device_mfu_percent': per_device_mfu_percent,
        'all_per_device_tflops': per_device_tflops,
        'per_device_tokens_per_second': avg_per_device_tokens_per_second,
        'per_device_mfu_percent': avg_per_device_mfu_percent,
        'per_device_tflops': avg_per_device_tflops,
        'all_global_tokens_per_second': global_tokens_per_second,
        'all_global_mfu_percent': global_mfu_percent,
        'all_global_tflops': global_tflops,
        'global_tokens_per_second': avg_global_tokens_per_second,
        'global_mfu_percent': avg_global_mfu_percent,
        'global_tflops': avg_global_tflops
    }


def process_ddp_log_files(log_files):
    """
    Process multiple log files and extract metrics.
    """
    results = {}
    data_count = 0
    for config, log_file in log_files.items():
        metrics = extract_ddp_metrics(log_file)
        if metrics:
            results[config] = metrics
            data_count += len(metrics['steps'])
            print(f"Processed {config}: {len(metrics['steps'])} data points found, world_size={metrics['world_size']}")
    return results, data_count


def plot_scaling_efficiency(results, output_dir="plots"):
    """
    Create plots showing scaling efficiency across different world sizes.
    """
    configs = list(results.keys())
    world_sizes = [results[c]['world_size'] for c in configs]
    approaches = {}
    # Group configurations by approach
    for config, metrics in results.items():
        parts = config.split('(')
        approach = parts[0].strip()
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append((config, metrics))
    
    # Create scaling plots for global tokens/sec
    plt.figure(figsize=(12, 8))
    for approach, configs_and_metrics in approaches.items():
        # Sort by world_size to ensure correct line plotting
        configs_and_metrics.sort(key=lambda x: x[1]['world_size'])
        x_values = [m['world_size'] for _, m in configs_and_metrics]
        y_values = [m['global_tokens_per_second'] for _, m in configs_and_metrics]
        # Calculate ideal scaling from the first point
        base_throughput = y_values[0]
        base_world_size = x_values[0]
        ideal_scaling = [base_throughput * (ws / base_world_size) for ws in x_values]
        # Plot both actual and ideal scaling
        plt.plot(x_values, y_values, 'o-', linewidth=2.5, label=f"{approach} (Actual)")
        # Only plot ideal line once if we have multiple approaches
        if approach == list(approaches.keys())[0]:
            plt.plot(x_values, ideal_scaling, 'k--', label="Ideal Linear Scaling")
    plt.title("Throughput Scaling with World Size", fontsize=18)
    plt.xlabel("World Size (Number of GPUs)", fontsize=14)
    plt.ylabel("Global Tokens per Second", fontsize=14)
    plt.xticks(sorted(list(set(world_sizes))))  # Use actual world sizes for x-axis
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add scaling efficiency for each point
    for approach, configs_and_metrics in approaches.items():
        configs_and_metrics.sort(key=lambda x: x[1]['world_size'])
        x_values = [m['world_size'] for _, m in configs_and_metrics]
        y_values = [m['global_tokens_per_second'] for _, m in configs_and_metrics]
        # Calculate scaling efficiency
        base_throughput = y_values[0]
        base_world_size = x_values[0]
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            if i == 0:  # Skip the first point (efficiency is 100%)
                continue
            ideal_y = base_throughput * (x / base_world_size)
            efficiency = (y / ideal_y) * 100
            plt.annotate(f"{efficiency:.1f}%", 
                        xy=(x, y), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        ha='left', va='bottom',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_throughput.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Create similar plot for Global MFU
    plt.figure(figsize=(12, 8))
    for approach, configs_and_metrics in approaches.items():
        configs_and_metrics.sort(key=lambda x: x[1]['world_size'])
        x_values = [m['world_size'] for _, m in configs_and_metrics]
        y_values = [m['global_mfu_percent'] for _, m in configs_and_metrics]
        plt.plot(x_values, y_values, 'o-', linewidth=2.5, label=approach)
    plt.title("Hardware Utilization Scaling with World Size", fontsize=18)
    plt.xlabel("World Size (Number of GPUs)", fontsize=14)
    plt.ylabel("Global Model FLOPS Utilization (%)", fontsize=14)
    plt.xticks(sorted(list(set(world_sizes))))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_mfu.png'), dpi=600, bbox_inches='tight')
    plt.close()


def extract_training_time_simple(log_path):
    """Extract start and end timestamps from a log file to calculate total training time"""
    with open(log_path, 'r') as f:
        content = f.read()
    # Find first step timestamp
    first_step_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Step: 1 \|', content)
    # Find last step (assuming 1000 steps total)
    last_step_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Step: 1000 \|', content)
    
    if first_step_match and last_step_match:
        first_time = datetime.datetime.strptime(first_step_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
        last_time = datetime.datetime.strptime(last_step_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
        total_seconds = (last_time - first_time).total_seconds()
        minutes = total_seconds / 60
        # Format for display
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.1f}"
        return {
            'total_seconds': total_seconds,
            'total_minutes': minutes,
            'formatted_time': formatted_time
        }
    return None


def plot_training_time_simple(results, output_dir="plots"):
    """Create plots showing training time across configurations"""        
    # Group by approach
    configs_with_timing = {c: m for c, m in results.items() if 'total_seconds' in m}
    approaches = {}
    for config, metrics in configs_with_timing.items():
        approach = config.split('(')[0].strip()
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append((config, metrics))
    
    # Plot training time
    plt.figure(figsize=(12, 8))
    for approach, config_metrics in approaches.items():
        # Sort by world size
        config_metrics.sort(key=lambda x: x[1]['world_size'])
        x_values = [m['world_size'] for _, m in config_metrics]
        y_values = [m['total_seconds'] for _, m in config_metrics]
        plt.plot(x_values, y_values, 'o-', linewidth=2.5, label=approach)
        # Add time labels
        for ws, mins, time_str in zip( x_values, y_values, [m['formatted_time'] for _, m in config_metrics]):
            plt.annotate(
                time_str,
                xy=(ws, mins),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
            )
    plt.title("Total Training Time (1000 steps)", fontsize=18)
    plt.xlabel("World Size (Number of GPUs)", fontsize=14)
    plt.ylabel("Total Time (seconds)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_training_time.png'), dpi=600)
    plt.close()
    
def plot_time_to_target_loss(results, target_loss=5.5, output_dir="plots"):
    """Plot time needed to reach a target loss value for each configuration"""
    plt.figure(figsize=(12, 8))
    # Group by approach
    approaches = {}
    for config, metrics in results.items():
        if 'total_seconds' not in metrics or 'loss' not in metrics or 'steps' not in metrics:
            continue
        approach = config.split('(')[0].strip()
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append((config, metrics))
    
    # For each approach, find time to reach target loss
    target_times = {}
    for approach, configs_and_metrics in approaches.items():
        target_times[approach] = []
        for config, metrics in configs_and_metrics:
            # Find step where loss drops below target
            steps = metrics['steps']
            losses = metrics['loss']
            world_size = metrics['world_size']
            time_per_step = metrics['total_seconds'] / 1000  # avg seconds per step
            for i, loss in enumerate(losses):
                if loss <= target_loss:
                    steps_needed = steps[i]
                    time_needed = steps_needed * time_per_step
                    target_times[approach].append((world_size, time_needed))
                    break
            else:
                # If target loss never reached, use max time
                target_times[approach].append((world_size, metrics['total_seconds']))
    
    # Plot time to target loss
    for approach, times in target_times.items():
        times.sort(key=lambda x: x[0])  # sort by world_size
        x_values = [t[0] for t in times]
        y_values = [t[1] for t in times]
        plt.plot(x_values, y_values, 'o-', linewidth=2.5, label=approach)
        # Annotate with efficiency vs. world_size=4
        base_time = y_values[0]
        base_ws = x_values[0]
        for i, (ws, time) in enumerate(zip(x_values, y_values)):
            if i == 0:
                continue
            speedup = base_time / time
            ideal_speedup = ws / base_ws
            efficiency = (speedup / ideal_speedup) * 100
            
            plt.annotate(f"{efficiency:.1f}%", 
                        xy=(ws, time), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        ha='left', va='bottom',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    plt.title(f"Time to Reach Loss ≤ {target_loss}", fontsize=18)
    plt.xlabel("World Size (Number of GPUs)", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_to_target_loss.png'), dpi=600)
    plt.close()


def plot_split_loss_curves(results, output_dir="plots"):
    """Create separate loss plots for padded and padding-free approaches"""
    # Group configurations by approach type
    padded_configs = {}
    paddingfree_configs = {}
    for config, metrics in results.items():
        if 'Padding-free' in config or 'token-list' in config:
            paddingfree_configs[config] = metrics
        else:
            padded_configs[config] = metrics
    # Create subplot for padded approaches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot padded approaches
    for config, metrics in padded_configs.items():
        ax1.plot(metrics['steps'], metrics['loss'], label=config)
    ax1.set_title("Loss Curves: Padded Approaches", fontsize=16)
    ax1.set_xlabel("Training Step", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=9)
    
    # Plot padding-free approaches
    for config, metrics in paddingfree_configs.items():
        ax2.plot(metrics['steps'], metrics['loss'], label=config)
    ax2.set_title("Loss Curves: Padding-Free Approaches", fontsize=16)
    ax2.set_xlabel("Training Step", fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=9)
    
    # Ensure both plots use the same y-axis
    y_min = min(
        min(min(m['loss']) for m in padded_configs.values()),
        min(min(m['loss']) for m in paddingfree_configs.values())
    )
    y_max = max(
        max(max(m['loss'][0:10]) for m in padded_configs.values()),
        max(max(m['loss'][0:10]) for m in paddingfree_configs.values())
    )
    ax1.set_ylim(y_min - 0.1, y_max + 0.1)
    ax2.set_ylim(y_min - 0.1, y_max + 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_loss_curves.png'), dpi=600)
    plt.close()


def plot_ddp_results(log_files, output_dir="plots"):
    """
    Generate and save visualization plots for DDP benchmarking.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Process log files
    results, data_count = process_ddp_log_files(log_files)
    if not results:
        print("No data found in log files. Exiting.")
        return
    
    # Print summary of results
    print(f"\nSummary of DDP Results:")
    print("-" * 120)
    print(f"{'Configuration':<35} {'World Size':<10} {'Global Tokens/sec':<20} {'Global MFU %':<15} {'Global TFLOPs':<15}")
    print("-" * 120)
    for config, metrics in results.items():
        print(f"{config:<35} {metrics['world_size']:<10} {metrics['global_tokens_per_second']:<20.2f} {metrics['global_mfu_percent']:<15.2f} {metrics['global_tflops']:<15.2f}")
    
    # Extract timing information
    for config, log_path in log_files.items():
        if config in results:
            timing_info = extract_training_time_simple(log_path)
            if timing_info:
                results[config].update(timing_info)
    
    # Create DDP-specific plots
    plot_scaling_efficiency(results, output_dir)
    plot_training_time_simple(results, output_dir)
    plot_time_to_target_loss(results, target_loss=5.5, output_dir=output_dir)
    plot_split_loss_curves(results, output_dir=output_dir)
    # Create line chart of loss over time for each configuration
    plot_time_series(results, 'loss', 'Validation Loss During Training', 'Loss', 'loss_over_time.png', output_dir)