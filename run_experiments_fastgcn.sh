#!/bin/bash

source ~/.bashrc
conda activate pygeo39

module load cuda11.3/toolkit/11.3.1
module load cuDNN/cuda11.1/8.0.5

# File to store results
method="fastgcn"
output_file1="results_table1.csv"
output_file2="results_table2.csv"

# Create the CSV files if it doesn't exist
if [ ! -f $output_file1 ]; then
    echo "Method,Dataset,Accuracy" > output_file1
fi

if [ ! -f $output_file2 ]; then
    echo "Method,Dataset,Sampling Number,Accuracy" > $output_file2
fi

# Define the list of datasets
datasets=("Cora")

# Define the list of sampling numbers
samp_nums=(32 64 128 256 512)

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Print multiple newlines and the dataset name
    echo -e "\n\nProcessing ${dataset}\n\n"

    # Experiment 1: Repeat the experiment 10 times and take the average accuracy
    python pytorch_ladies.py --cuda 0 --dataset "${dataset,,}" --runs 3 > ${dataset}_${method}_exp1.output

    # Extract accuracy from the output (assuming it's in a format like "Accuracy: xx.xx%")
    accuracy=$(grep -Eo "Mini Acc: [0-9.]+ ± [0-9.]+" ${dataset}_${method}_exp1.output)

    # Append the dataset, sampling number, and accuracy to the CSV file
    echo "$method,$dataset,$accuracy" >> $output_file1

    # Experiment 2: Iterate over each sampling number

    for samp_num in "${samp_nums[@]}"; do
        # Execute the command with the current dataset and sampling number and capture the full output
        python3 pytorch_ladies.py --cuda 0 --dataset "${dataset,,}" --samp_num "$samp_num" > ${dataset}_${method}_exp2.output

        # Extract accuracy from the output (assuming it's in a format like "Accuracy: xx.xx%")
        accuracy=$(grep -Eo "Mini Acc: [0-9.]+ " ${dataset}_${method}_exp2.output | awk '{print $3}')

        # Append the dataset, sampling number, and accuracy to the CSV file
        echo "$method,$dataset,$samp_num,$accuracy" >> $output_file2
    done
done


