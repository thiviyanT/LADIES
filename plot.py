import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('science')

# Read data from CSV
dataframe = pd.read_csv("results.csv")

# Create a dictionary from the dataframe
data = {}
for index, row in dataframe.iterrows():
    model = row['Method']
    dataset = row['Dataset']
    sample = row['Sampling Number']
    accuracy = float(row['Accuracy'])

    if model not in data:
        data[model] = {}
    if dataset not in data[model]:
        data[model][dataset] = {}

    data[model][dataset][sample] = accuracy

# Colors for models
colors = {
    'Ladies': 'red',
    'FastGCN': 'green',
    'GAS': 'blue',
    'GraphSAINT': 'orange'
}

# Placeholder for 'All'
all_placeholder = 700

unique_datasets = set()
for _, datasets in data.items():
    for dataset in datasets.keys():
        unique_datasets.add(dataset)

# Determine the number of rows and columns for subplots
n_datasets = len(unique_datasets)
n_cols = 4
n_rows = n_datasets // n_cols
if n_datasets % n_cols:
    n_rows += 1

with plt.style.context('science'):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Ensure axes is always a 2D array, even if there's only one subplot
    if n_datasets == 1:
        axes = np.array([axes])

    for ax, dataset in zip(axes.flatten(), unique_datasets):
        for model, datasets in data.items():
            if dataset in datasets:
                x = []
                y = []
                for sample, accuracy in datasets[dataset].items():
                    if sample == 'All':
                        sample = all_placeholder
                    else:
                        sample = int(sample)
                    x.append(sample)
                    y.append(accuracy)
                ax.plot(x, y, label=model, marker='o', color=colors[model])

        ax.set_title(f"{dataset}", fontsize=18)
        # ax.set_xlabel('Number of Samples')
        # ax.set_ylabel('Classification Accuracy (%)')
        ax.legend()
        ax.grid(True)
        ax.set_xticks([32, 64, 128, 256, 512, all_placeholder])
        ax.set_xticklabels([32, 64, 128, 256, 512, 'All'], rotation=90, fontsize=10)
        ax.set_ylim(0, 100)  # Set y-axis limits

    # Remove any extra unused subplots
    for i in range(n_datasets, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.savefig("sampling_vs_accuracy.pdf", format='pdf', bbox_inches='tight')
    plt.show()
