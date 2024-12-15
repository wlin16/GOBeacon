# GOBeacon: Ensemble Model for Protein Function Prediction

GOBeacon is an advanced ensemble model that integrates structure-aware protein language model embeddings with protein-protein interaction networks to predict protein functions with high accuracy. Utilizing a contrastive learning framework, GOBeacon significantly outperforms existing methods on various benchmarks.

## Key Features

- **Multi-Modality Integration:** Combines embeddings from protein language models and graph data from protein-protein interactions.
- **High Accuracy:** Achieves high Fmax scores across multiple categories (BP, MF, CC) on the CAFA3 benchmark.
- **Versatility:** Effective in structure-based function prediction tasks, matching or exceeding specialized tools.

## Getting Started

### 1. Clone the code repo into your own dir
```bash
git clone git@github.com:wlin16/GOBeacon.git
cd GOBeacon
```

### 2. Prepare the Environment
install all required Python packages:

Ensure you have Python installed on your system. You can install all required packages using:

```bash
pip install -r requirements.txt
```

### 3. Data Preparation
Navigate to the `run_scripts` directory and execute the data mapping script. Perform this step for each benchmark dataset and its corresponding training and test set to ensure all necessary data is prepared:
```bash
cd run_scripts
# Define the path and run the script for each benchmark dataset and its corresponding training and test set
bash string_network_mapping.sh
```

### 4. Prepare Embedding and Graph Data
Still in the run_scripts directory, execute the feature extraction script. Remember to specify the `PPI_PATH` and generate data for BP, MF, and CC.
```bash
bash feat_extract.sh
# Run this for each benchmark dataset and for both training and test sets
```

## Train the models
After preparing the embeddings, navigate back to the run_scripts directory. Run the training script for each ontology (`BP`, `MF`, `CC`). This means that eventually you should have `9` trained model for all sub-ontology!!! It is crucial to train three modality models for each ontology to harness the full predictive power of GOBeacon:
```bash
cd run_scripts
# Define the path and run the training script for each ontology (BP, MF, CC)
bash train.sh
```

## Prediction
Once the models are trained for each sub-ontology, execute the prediction script for each one. This step is essential to generate predictions from the trained models and should be done for each ontology:
```bash
cd run_scripts
# Define the path and execute the prediction script
bash predict.sh
```



