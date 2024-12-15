ONTOLOGY=CC # BP, CC, or MF
BENCHMARK=CAFA3 # CAFA3 or PDBch
CSV_PATH=./data # put where you save your csv
TARGET_CSV=test_df.pkl # put your csv's name

ROOT_DIR=/path/to/your/project/dir # ! Need to specify!!!!!!! e.g. ~/GOBeacon
cd $ROOT_DIR

# 1. infer prediction using seq model
INFER_TYPE=sequence
SEQ_LMDB_DIR=data/lmdb/${BENCHMARK}_${INFER_TYPE}
SEQ_MODEL_CKPT=./result/${BENCHMARK}/model/${ONTOLOGY}_sequence.ckpt
RESULT_SAVE_FILE=result/${BENCHMARK}/${ONTOLOGY}_seq.pt
MODEL_INPUT_DIM=1536
python3 main.py \
        general.usage=infer \
        dataset.load_data.sub_ontology=$ONTOLOGY \
        dataset.load_data.lmdb_path=$SEQ_LMDB_DIR \
        predict.model_ckpt=$SEQ_MODEL_CKPT \
        model.num_features=$MODEL_INPUT_DIM \
        predict.save_name=$RESULT_SAVE_FILE


# 2. infer prediction using struct model
INFER_TYPE=structure
STRUCT_LMDB_DIR=data/lmdb/${BENCHMARK}_${INFER_TYPE}
STRUCT_MODEL_CKPT=./result/${BENCHMARK}/model/${ONTOLOGY}_structure.ckpt
RESULT_SAVE_FILE=./result/${BENCHMARK}/${ONTOLOGY}_struct.pt
MODEL_INPUT_DIM=1024
python3 main.py \
        general.usage=infer \
        dataset.load_data.sub_ontology=$ONTOLOGY \
        dataset.load_data.lmdb_path=$STRUCT_LMDB_DIR \
        predict.model_ckpt=$STRUCT_MODEL_CKPT \
        model.num_features=$MODEL_INPUT_DIM \
        predict.save_name=$RESULT_SAVE_FILE

# 3. infer prediction using graph model
INFER_TYPE=graph
GRAPH_LMDB_DIR=data/lmdb/${BENCHMARK}_${INFER_TYPE}
GRAPH_MODEL_CKPT=./result/${BENCHMARK}/model/${ONTOLOGY}_graph.ckpt
RESULT_SAVE_FILE=result/${BENCHMARK}/${ONTOLOGY}_graph.pt
MODEL_INPUT_DIM=1024
python3 main.py \
        general.usage=infer \
        model.model_choice=graph \
        dataset.load_data.sub_ontology=$ONTOLOGY \
        dataset.load_data.lmdb_path=$GRAPH_LMDB_DIR \
        predict.model_ckpt=$GRAPH_MODEL_CKPT \
        model.num_features=$MODEL_INPUT_DIM \
        predict.save_name=$RESULT_SAVE_FILE


# # 4. Finally, calculate the Fmax score according to the ensemble result
python3 main.py \
        general.usage=predict \
        predict.ensemble_dir=./result/${BENCHMARK}/ \
        dataset.load_data.ds_path=$CSV_PATH \
        dataset.load_data.target_centre_node_csv=$TARGET_CSV \
        dataset.load_data.sub_ontology=$ONTOLOGY