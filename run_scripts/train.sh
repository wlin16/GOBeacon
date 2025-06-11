#!/bin/bash

# Set the ontology type, benchmark dataset, LMDB directory, model choice, and training type
ONTOLOGY=CC # BP, CC, or MF
BENCHMARK=CAFA3 # CAFA3 or PDBch
TRAIN_TYPE=structure # sequence, structure or graph
LMDB_DIR=data/lmdb/${BENCHMARK}_${TRAIN_TYPE}
MODEL_SAVE_PATH=./result/${BENCHMARK}/model
MODEL_FILENAME=${ONTOLOGY}_${TRAIN_TYPE}

# Determine the input dimension and the architecture of the model based on the training type
if [ "$TRAIN_TYPE" = "sequence" ]; then
    MODEL_INPUT_DIM=1280
    MODEL_CHOICE=mlp
elif [ "$TRAIN_TYPE" = "structure" ]; then
    MODEL_INPUT_DIM=1024
    MODEL_CHOICE=mlp
elif [ "$TRAIN_TYPE" = "graph" ]; then
    MODEL_INPUT_DIM=1024
    MODEL_CHOICE=graph
fi

# Navigate to the root directory of the project repository
ROOT_DIR=/path/to/your/project/dir # ! Need to specify!!!!!!! e.g. ~/GOBeacon
cd $ROOT_DIR

# Execute the main Python script with appropriate command-line arguments for training
python3 main.py \
        general.usage=train \
        dataset.load_data.sub_ontology=$ONTOLOGY \
        model.model_choice=$MODEL_CHOICE \
        model.train_type=$TRAIN_TYPE \
        model.num_features=$MODEL_INPUT_DIM \
        dataset.load_data.lmdb_path=$LMDB_DIR \
        model.model_save_path=$MODEL_SAVE_PATH \
        model.model_save_filename=$MODEL_FILENAME

# Description: This script is configured to train models based on specified training types and model choices.
# It dynamically adjusts model input dimensions according to the type of training data (sequence, structure, or graph).
# Usage:
#   Ensure the script has executable permissions: chmod +x train.sh
#   Run the script by entering: ./train.sh