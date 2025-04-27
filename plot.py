import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define file paths (assuming all files are inside 'output/' directory)
input_kernel_file = 'output/cnn_in_execution_times.txt'
torso_kernel_file = 'output/cnn_torso_execution_times.txt'
output_kernel_file = 'output/cnn_out_execution_times.txt'

# Load datasets
datasets = [
    pd.read_csv(input_kernel_file, delim_whitespace=True, header=None),
    pd.read_csv(torso_kernel_file, delim_whitespace=True, header=None),
    pd.read_csv(output_kernel_file, delim_whitespace=True, header=None)
]

titles = ['Input Kernel Execution Times', 'Torso Kernel Execution Times', 'Output Kernel Execution Times']
column_labels = ['CPU', 'GPU', 'Parallel GPU']

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

speedups_vs_gpu = []
speedups_vs_cpu = []

for i, (data, title) in enumerate(zip(datasets, titles)):
    # Exclude the first row (outlier) when calculating mean/min
    data_filtered = data.iloc[1:]

    means = data_filtered.mean()
    mins = data_filtered.min()
    maxs = data.max()  # Include outlier for max values

    # Compute speedups
    parallel_gpu = data_filtered[2]
    gpu = data_filtered[1]
    cpu = data_filtered[0]

    speedup_vs_gpu = (gpu / parallel_gpu).mean()
    speedup_vs_cpu = (cpu / parallel_gpu).mean()

    speedups_vs_gpu.append(speedup_vs_gpu)
    speedups_vs_cpu.append(speedup_vs_cpu)

    # Plot bars
    x = np.arange(3)
    axs[i].bar(x, means, width=0.5, tick_label=column_labels)

    # Plot min-max lines
    for xi, min_val, max_val in zip(x, mins, maxs):
        axs[i].plot([xi, xi], [min_val, max_val], color='r', linestyle='--', marker='o')

    axs[i].set_yscale('log')
    axs[i].set_title(title)
    axs[i].set_ylabel('Execution Time (ms)')
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(column_labels, rotation=15, ha='right')
    axs[i].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[i].legend(['Min-Max Range'])

plt.tight_layout()
plt.show()

# Print speedup summary
speedup_summary = pd.DataFrame({
    'Component': ['Input Kernel', 'Torso Kernel', 'Output Kernel'],
    'Speedup vs GPU': speedups_vs_gpu,
    'Speedup vs CPU': speedups_vs_cpu
})

print(speedup_summary)
