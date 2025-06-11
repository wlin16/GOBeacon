#!/bin/bash

# Description: This script is used to process STRING database interactions for specific datasets.
# It accepts three parameters: the name of the benchmark dataset (CAFA3 or PDBch), the type of dataset (train or test), and the output directory for the results.
# Usage:
#   bash string_network_mapping.sh 

ROOT_DIR=/path/to/your/project/dir # ! Need to specify!!!!!!! e.g. ~/GOBeacon

cd $ROOT_DIR

# Example1:
# Get PPI network from STRING for CAFA3 training set:
BENCHMARK=CAFA3 # CAFA3 or PDBch
FILENAME=test # train_df.pkl or test_df.pkl
# PPI_OUTPUT_DIR=./string_training_crawl
PPI_OUTPUT_DIR=/path/to/your/ppi_data # ! Need to specify!!!!!!! e.g. ~/GOBeacon/data/string_training_crawl
python3 mapping.py -b $BENCHMARK -d $FILENAME -o $PPI_OUTPUT_DIR

# # Example2:
# # Get PPI network from STRING for PDBch test set:
# BENCHMARK=PDBch # CAFA3 or PDBch
# DATASET=test # train or test
# PPI_OUTPUT_DIR=./string_training_crawl
# python3 mapping.py -b $BENCHMARK -d $DATASET -o $PPI_OUTPUT_DIR