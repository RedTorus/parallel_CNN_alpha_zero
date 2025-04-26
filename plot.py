import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File path
file_path = Path("output/model_execution_times.txt")

# Load data
data = np.loadtxt(file_path)
cpu_times = data[:, 0]
gpu_default_times = data[:, 1]
gpu_custom_times = data[:, 2]

# Remove first row for averages (outlier)
cpu_avg = np.mean(cpu_times[1:])
gpu_default_avg = np.mean(gpu_default_times[1:])
gpu_custom_avg = np.mean(gpu_custom_times[1:])

cpu_max, cpu_min = np.max(cpu_times), np.min(cpu_times)
gpu_default_max, gpu_default_min = np.max(gpu_default_times), np.min(gpu_default_times)
gpu_custom_max, gpu_custom_min = np.max(gpu_custom_times), np.min(gpu_custom_times)

# Speedups
speedup_custom_vs_cpu = cpu_times / gpu_custom_times
speedup_custom_vs_default_gpu = gpu_default_times / gpu_custom_times

# Remove first speedup entry for average computation
speedup_custom_vs_cpu_avg = np.mean(speedup_custom_vs_cpu[1:])
speedup_custom_vs_cpu_max = np.max(speedup_custom_vs_cpu)

speedup_custom_vs_default_gpu_avg = np.mean(speedup_custom_vs_default_gpu[1:])
speedup_custom_vs_default_gpu_max = np.max(speedup_custom_vs_default_gpu)

# Print speedup results
print(f"Average speedup (Custom GPU vs CPU): {speedup_custom_vs_cpu_avg:.2f}x")
print(f"Maximum speedup (Custom GPU vs CPU): {speedup_custom_vs_cpu_max:.2f}x")
print(f"Average speedup (Custom GPU vs Default GPU): {speedup_custom_vs_default_gpu_avg:.2f}x")
print(f"Maximum speedup (Custom GPU vs Default GPU): {speedup_custom_vs_default_gpu_max:.2f}x")

# Plotting
labels = ['Avg', 'Max', 'Min']
cpu_vals = [cpu_avg, cpu_max, cpu_min]
gpu_default_vals = [gpu_default_avg, gpu_default_max, gpu_default_min]
gpu_custom_vals = [gpu_custom_avg, gpu_custom_max, gpu_custom_min]

x = np.arange(len(labels))  # [0, 1, 2]
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - bar_width, cpu_vals, width=bar_width, label='CPU', color='#1f77b4')
ax.bar(x, gpu_default_vals, width=bar_width, label='Default GPU', color='#ff7f0e')
ax.bar(x + bar_width, gpu_custom_vals, width=bar_width, label='Custom GPU', color='#2ca02c')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Execution Time (ms)")
ax.set_title("Execution Time Comparison")
ax.set_yscale('log')
ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)
ax.legend()

plt.tight_layout()
plt.show()
