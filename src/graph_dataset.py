import networkx as nx
import pickle as pkl
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
import pickle as pkl
import numpy as np
from src.lmdb_loader import LMDBLoader
import os
from src.hetroData import HetData


class Graph_Dataset(Dataset):
    # mode in ['train', 'valid', 'infer']
    def __init__(self, config, mode="train", key_list=None):
        self.data_dir = config.ds_path
        self.sub_ontology = config.sub_ontology

        self.mode = mode
        
        if self.mode in ['train', 'valid']:
            self.csv_path = config.train_centre_node_csv
        elif self.mode == 'test':
            self.csv_path = config.test_centre_node_csv
        elif self.mode == 'infer':
            self.csv_path = config.target_centre_node_csv
        else:
            raise ValueError("Invalid mode specified")

        self.train_df = pd.read_pickle(os.path.join(
            self.data_dir, config.benchmark, config.train_centre_node_csv))[config.sub_ontology]
        
        self.label_list = self.train_df.loc['all']
        self.label_list = np.array(self.label_list)

        self.df = pd.read_pickle(os.path.join(
            self.data_dir, config.benchmark, self.csv_path))[config.sub_ontology]
        
        self.key_list = self.df.index if self.mode in [
            'test', 'infer'] else key_list

        self.dataset = LMDBLoader(config.lmdb_path, self.key_list)

        """
        self.same_protein_dict -> k:prot v: [label, prot_w_label]
        self.para_protein_dict -> k:prot+label v: [label, prot_wo_label_w_paralabel, paralabel]
        self.go_negative_prot_dict -> k: label v: [prot_wo_label_wo_shared_labelupstream]
        """
        if mode == "train":
            with open(os.path.join(self.data_dir, config.prot_relation_file[config.sub_ontology]), 'rb') as f:
                self.same_protein_dict, self.para_protein_dict, self.go_negative_prot_dict = pkl.load(
                    f)

    def label_to_ytrue(self, labels):
        true_label =(self.label_list == np.array(labels)[:, None]).sum(0).astype(np.float32)

        return true_label

    def label_to_idx(self, label):

        label_idx_array = np.where(
            (self.label_list == np.array([label])) == 1)[0]
        return label_idx_array
    
    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        """
        return one anchor sample, one positive sample, one para sample and one negative sample
        """
        anchor_data = self.dataset[idx]
        anchor_seqname = self.key_list[idx]

        if self.mode == 'train':
            if self.df.loc[anchor_seqname] == []:
                true_y = np.zeros([len(self.label_list)])
            else:
                true_y = self.label_to_ytrue(self.df.loc[anchor_seqname])

            true_y = torch.from_numpy(true_y).to(torch.float32)

            sample = HetData(anchor_data.x, 
                            anchor_data.edge_index, 
                            anchor_data.edge_attr,
                            y_true=true_y, 
                            pid=anchor_seqname, 
                            mode = self.mode)

        elif self.mode != "train":
            if self.df.loc[anchor_seqname] == []:
                true_y = np.zeros([len(self.label_list)])
            else:
                true_y = self.label_to_ytrue(self.df.loc[anchor_seqname])
            true_y = torch.from_numpy(true_y).to(torch.float32)

            sample = HetData(anchor_data.x, 
                            anchor_data.edge_index, 
                            anchor_data.edge_attr,
                            y_true=true_y, 
                            pid=anchor_seqname, 
                            mode = self.mode)

        return sample 
