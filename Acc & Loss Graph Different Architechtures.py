import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all datasets
data_files = {
    '40': '/Users/OliverBenton/Repositories/1DMNIST/Poster Data/40_1_layer.csv',
    '80': '/Users/OliverBenton/Repositories/1DMNIST/Poster Data/80_2_layer.csv',
    '160': '/Users/OliverBenton/Repositories/1DMNIST/Poster Data/160_3_layer.csv',
    '320': '/Users/OliverBenton/Repositories/1DMNIST/Poster Data/320_4_layer.csv',
}

datasets = {}
for name, path in data_files.items():
    datasets[name] = pd.read_csv(path)

# Define the metrics and methods
metrics = {
    'Accuracy': {
        'No reg': 'Test acc (No reg)',
        'L2': 'Test acc (L2)',
        'Glial_pr': 'Test acc (Glial_pr)',
        'Glial_p': 'Test acc (Glial_p)'
    },
    'Loss': {
        'No reg': 'Test loss (No reg)',
        'L2': 'Test loss (L2)',
        'Glial_pr': 'Test loss (Glial_pr)',
        'Glial_p': 'Test loss (Glial_p)'
    }
}

methods = list(metrics['Accuracy'].keys())
epoch_counts = list(datasets.keys())

# Create DataFrames to store calculated values
results = []

# Calculate statistics and store in results
for metric_name, metric_cols in metrics.items():
    for method, col in metric_cols.items():
        for epoch, data in datasets.items():
            values = data[col]
            mean = np.mean(values)
            std = np.std(values)
            min_val = mean - std
            max_val = mean + std
            
            results.append({
                'Metric': metric_name,
                'Method': method,
                'Epochs': epoch,
                'Mean': mean,
                'Std': std,
                'Min': min_val,
                'Max': max_val
            })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('/Users/OliverBenton/Repositories/1DMNIST/Poster Data/one_layer_size_architecture_comparison_stats.csv', index=False)
print("Statistics saved to 'regression_comparison_stats.csv'")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Regularization Method Comparison Across 1 layer Architectures (Differential Hyperparameters)', fontsize=16, y=0.97)

# Set up bar positions
bar_width = 0.2
x_pos = np.arange(len(epoch_counts))

# Colors for different methods
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot Accuracy
for i, method in enumerate(methods):
    method_data = results_df[(results_df['Metric'] == 'Accuracy') & (results_df['Method'] == method)]
    means = method_data['Mean'].values
    stds = method_data['Std'].values
    
    ax1.bar(x_pos + i*bar_width, means, width=bar_width,
            yerr=stds, label=method, color=colors[i],
            align='center', alpha=0.8, ecolor='black', capsize=5)

ax1.set_title('Test Accuracy Comparison', pad=15)
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1.1)
ax1.set_xticks(x_pos + (len(methods)-1)*bar_width/2)
ax1.set_xticklabels(epoch_counts)
ax1.set_xlabel('Size of hidden layer(neurons) ')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.4)

# Plot Loss
for i, method in enumerate(methods):
    method_data = results_df[(results_df['Metric'] == 'Loss') & (results_df['Method'] == method)]
    means = method_data['Mean'].values
    stds = method_data['Std'].values
    
    ax2.bar(x_pos + i*bar_width, means, width=bar_width,
            yerr=stds, label=method, color=colors[i],
            align='center', alpha=0.8, ecolor='black', capsize=5)

ax2.set_title('Test Loss Comparison', pad=15)
ax2.set_ylabel('Loss')
ax2.set_xticks(x_pos + (len(methods)-1)*bar_width/2)
ax2.set_xticklabels(epoch_counts)
ax2.set_xlabel('Size of hidden layer(neurons)')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
#plt.savefig(f"/Users/OliverBenton/Repositories/1DMNIST/Poster data/one_layer_size_architecture_comparison_plot.png")
plt.show()