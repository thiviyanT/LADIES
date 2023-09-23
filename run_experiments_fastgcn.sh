#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=fastgcn
#SBATCH --cpus-per-task=10
#SBATCH --output=fastgcn_output.%j.out
#SBATCH --error=fastgcn_error.%j.err

# Load any modules or source bashrc if necessary
source ~/.bashrc
conda activate pygeo45

module load cuda11.3/toolkit/11.3.1
module load cuDNN/cuda11.1/8.0.5

# File to store results
method="fastgcn"
output_file1="results_table1.csv"
output_file2="results_table2.csv"

## Create the CSV files if they don't exist
#if [ ! -f $output_file1 ]; then
#    echo "Method,Dataset,Accuracy(mean),Accuracy(std)" > $output_file1
#fi

if [ ! -f $output_file2 ]; then
    echo "Method,Dataset,Sampling Number,Accuracy(mean),Accuracy(std)" > $output_file2
fi

# Define the list of datasets
datasets=("Cora" "CiteSeer" "PubMed" "Reddit" "Yelp" "Flickr" "arxiv" "products")

# Define the list of sampling numbers
samp_nums=(32 64 128 256 512)

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    echo -e "\n\nProcessing ${dataset}\n\n"

#    # Experiment 1 - Repeat 10 times
#    python pytorch_ladies.py --cuda 0 --dataset "${dataset,,}" --runs 10 --sample_method ${method} --data_dir /var/scratch/tsingam/ > ${dataset}_${method}_exp1.output
#
#    mean=$(grep -Eo "Mini Acc: [0-9.]+ ± [0-9.]+" ${dataset}_${method}_exp1.output | awk '{print $3}')
#    std=$(grep -Eo "Mini Acc: [0-9.]+ ± [0-9.]+" ${dataset}_${method}_exp1.output | awk '{print $5}')
#    echo "$method,$dataset,$mean,$std" >> $output_file1

    # Experiment 2 - Repeat 5 times
    for samp_num in "${samp_nums[@]}"; do
        python3 pytorch_ladies.py --cuda 0 --dataset "${dataset,,}" --runs 5 --sample_method ${method} --samp_num "$samp_num" --data_dir /var/scratch/tsingam/ > ${dataset}_${method}_exp2.output

        mean=$(grep -Eo "Mini Acc: [0-9.]+ ± [0-9.]+" ${dataset}_${method}_exp2.output | awk '{print $3}')
        std=$(grep -Eo "Mini Acc: [0-9.]+ ± [0-9.]+" ${dataset}_${method}_exp2.output | awk '{print $5}')
        echo "$method,$dataset,,$samp_num,$mean,$std" >> $output_file2
    done
done
