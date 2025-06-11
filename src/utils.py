
import os
import pickle
import pandas as pd
import numpy as np
import ast
import lmdb
import pickle


def check_exists(path):
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist.")


def sub_ontology_label_num(cfg):
    path = os.path.join(cfg.ds_path, cfg.bottom_label_file)
    check_exists(path)
    sub_ontology_dict = pickle.load(open(path, "rb"))[cfg.sub_ontology]
    go_terms_set = set()
    for terms in sub_ontology_dict.values():
        go_terms_set.update(terms)
    return go_terms_set

def create_folder(folder_name):
    os.makedirs(os.path.dirname(folder_name), exist_ok=True)

def convert_go_terms_to_indices(row, go_index_dict):
    return [go_index_dict.get(go_term) for go_term in row if go_term in go_index_dict]


def read_df(cfg, pid_list=None, mode="train"):

    mode_dict = {
        "train": cfg.train_centre_node_csv,
        "test": cfg.test_centre_node_csv,
        "target": cfg.target_centre_node_csv
    }

    csv_file = mode_dict[mode]
    df_path = os.path.join(cfg.ds_path,  cfg.benchmark, csv_file)
    check_exists(df_path)
    full_df_path = os.path.join(cfg.ds_path, cfg.benchmark, cfg.train_centre_node_csv)
    df = pd.read_pickle(full_df_path)[cfg.sub_ontology]
    if mode == "test" or "target":
        input_df = pd.read_pickle(df_path)[cfg.sub_ontology]
        if "all" in input_df.index:
            input_df.pop("all")
    else:
        input_df = df[:-1]
    terms = df.loc['all']
    go_terms_index = {go_term: index for index, go_term in enumerate(terms)}

    if pid_list is not None:
        pid_list = read_txt(cfg, pid_list)
        input_df = input_df[input_df.apply(lambda x: x != [])]
        input_df = input_df[input_df.index.isin(pid_list)]
    PID_list = input_df.index.tolist()
    
    input_df = input_df.reset_index()
    input_df[cfg.sub_ontology] = input_df[cfg.sub_ontology].apply(lambda row: convert_go_terms_to_indices(row, go_terms_index))
    indexed_annotations = input_df[cfg.sub_ontology].tolist()

    return PID_list, indexed_annotations


def read_txt(cfg, file_name):
    txt_path = os.path.join(cfg.ds_path, cfg.benchmark, file_name)
    with open(txt_path, 'r') as file:
        pid_list = [line.strip() for line in file]
        return pid_list


def initialize(lmdb_path):
    env = lmdb.open(lmdb_path,
                    map_size=200 * (1024 * 1024 * 1024),  # 200G
                    create=True,
                    subdir=True,
                    readonly=False)
    return env


def insert_data(env, key, value):
    txn = env.begin(write=True)
    txn.put(key, value)
    txn.commit()


def delete(env, sid):
    txn = env.begin(write=True)
    txn.delete(str(sid).encode())
    txn.commit()


def read_data(env, name):
    '''
    e.g. 
    env = lmdb.open('/mnt/hdd16/weinilin/GO_pred_data/lmdb/test', readonly=True)
    read_data(env,"B6ZI38")
    env.close()
    '''
    txn = env.begin(write=False)
    data = txn.get(name.encode())
    print(pickle.loads(data))


def return_lmdb_keys(env):
    with env.begin() as txn:
        cursor = txn.cursor()
        keys = []
        for key, value in cursor:
            keys.append(key.decode('utf-8'))
    return keys
