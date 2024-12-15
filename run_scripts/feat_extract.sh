#!/bin/bash

# Description: This script is designed for extracting embeddings from different types of data related to protein-protein interactions (PPI) for specific datasets. It supports multiple types of data extraction such as graph, sequence, or structure-based embeddings depending on the specified requirements.
# It sets up the necessary environment and paths, and then runs a Python script with several parameters that define how and where to extract the features from.

# Change to the root directory where the project repository is located.
ROOT_DIR=/path/to/your/project/dir # ! Need to specify!!!!!!! e.g. ~/GOBeacon
cd $ROOT_DIR

# Define the benchmark dataset (options: CAFA3 or PDBch). This parameter determines which dataset to process.
BENCHMARK=PDBch

# Define the specific dataset file (options: train_df.pkl or test_df.pkl).
DATASET=train_df.pkl

# Specify the directory where the PPI data is stored.
PPI_PATH=/path/to/your/ppi_data # ! Need to specify!!!!!!! e.g. ~/GOBeacon/data/string_training_crawl

# Define the type of data extraction to perform (options: graph, sequence, structure). This parameter dictates the format of the embeddings to be extracted.
EXTRACT_TYPE=graph

# Set the output directory where the extracted features will be saved. This is organised by benchmark and extraction type.
OUTPUT_DIR=data/lmdb/${BENCHMARK}_${EXTRACT_TYPE}

# Execute the main Python script with the configuration specified by command-line arguments. This script utilizes the 'hydra' configuration library to manage complex configurations dynamically.
python3 main.py general.usage=feat_extract \
        dataset.feat_extract.benchmark=${BENCHMARK} \
        dataset.feat_extract.centre_node_csv=${DATASET} \
        dataset.feat_extract.ppi_path=${PPI_PATH} \
        dataset.feat_extract.extract_type=${EXTRACT_TYPE} \
        dataset.feat_extract.lmdb_path=$OUTPUT_DIR

# Usage:
# Ensure that the script has executable permissions: chmod +x feat_extract.sh
# Run the script by typing in the terminal: ./feat_extract.sh
# Note: Modify the parameters as needed based on your specific dataset and desired extraction type.