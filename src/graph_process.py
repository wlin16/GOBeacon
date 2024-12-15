#/$$
# $ @Author       : Weining Lin
# $ @Date         : 2024-04-17 01:58
# $ @LastEditTime : 2024-04-17 02:25
#$/
import src.utils as utils
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from src.lmdb_loader import LMDBLoader

import re
import logging
from tqdm import tqdm
import pickle
import os


class GraphGenerator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def build_sparse_matrix(self, df, node_map):
        src_indices = df['protein1'].map(node_map).values
        dst_indices = df['protein2'].map(node_map).values
        values = df['score'].values

        adjacency_matrix = sp.coo_matrix(
            (values, (src_indices, dst_indices)), shape=(len(node_map), len(node_map)))
        return adjacency_matrix.tocsr()

    def create_graph(self, df, node_map, node_features, is_orphan=False):
        if is_orphan:
            src_indices = [0]
            dst_indices = [0]
            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long)
            edge_attr = torch.tensor([1.0], dtype=torch.float)  
            x = node_features[df.protein.tolist()[0]]
            x = x.unsqueeze(1)
        else:
            src_indices = df['protein1'].map(node_map).values
            dst_indices = df['protein2'].map(node_map).values
            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long)

            edge_attr = torch.tensor(df['score'].values, dtype=torch.float)

            unique_nodes = sorted(node_map.keys(), key=lambda k: node_map[k])

            node_features_list = [node_features[node] for node in unique_nodes]
            x = torch.stack(node_features_list, dim=0)

        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr)
        return data
    
    def graph_generator(self, ppi_csv_path, centre_node_csv):
        centre_df = pd.read_pickle(centre_node_csv)

        if "all" in centre_df.index:
            centre_df = centre_df.drop(index = "all")
        centre_df.index.name = 'PID'
        centre_df.reset_index(inplace=True)

        center_node_list = centre_df[~centre_df.STRING.isna(
        )].PID.tolist()
        orphan_node_list = centre_df[centre_df.STRING.isna(
        )].PID.tolist()

        logging.info(f"Generating graphs for dataframe: [{centre_node_csv}]")

        logging.info(
            f'There are {len(center_node_list)} nodes have PPI information.')
        logging.info(
            f'There are {len(orphan_node_list)} nodes are orphan proteins.')
        
        model, tokenizer = self._load_model_and_tokenizer()
        embedder = self._ProstT5_emb_fetch

        lmdb_path = self.cfg.lmdb_path
        os.makedirs(lmdb_path, exist_ok=True)
        logging.info(f"Saving graphs to {lmdb_path}")
        env = utils.initialize(lmdb_path)

        logging.info(
            f'Generating graphs for {len(center_node_list)} proteins...')
        
        for centre_node_pid in tqdm(center_node_list, total=len(center_node_list)):
            dtype_spec = {
                'protein1': 'str',
                'protein2': 'str' 
            }
            ppi_csv = pd.read_csv(os.path.join(
                ppi_csv_path, centre_node_pid + '.csv'), dtype=dtype_spec)
            ppi_df = ppi_csv.sort_values(by=['score'], ascending=False)[
                :self.cfg.top]

            flat_ppi_df = pd.concat([
                ppi_df[['protein1', 'seq1']].rename(
                    columns={'protein1': 'protein', 'seq1': 'seq'}),
                ppi_df[['protein2', 'seq2']].rename(
                    columns={'protein2': 'protein', 'seq2': 'seq'})
            ])
            flat_ppi_df = flat_ppi_df.drop_duplicates(
                subset=['protein'], keep="first")
            
            string_centre_node = centre_df.loc[centre_df.PID == centre_node_pid, 'STRING'].iloc[0]

            unique_nodes = [string_centre_node] + list(flat_ppi_df.loc[flat_ppi_df.protein != string_centre_node, 'protein'].unique())
            node_map = {node: i for i, node in enumerate(unique_nodes)}

            node_features = embedder(model, tokenizer, flat_ppi_df)  # {"pid": array(1,1024)}
            graph = self.create_graph(ppi_df, node_map, node_features)

            utils.insert_data(env, centre_node_pid.encode(),
                              pickle.dumps(graph))

        logging.info(f'Generating self-loop graphs for Orphan proteins...')

        for orphan_node_pid in tqdm(orphan_node_list, total=len(orphan_node_list)):
            node_map = {orphan_node_pid: 0}
            node_df = centre_df[centre_df.PID == orphan_node_pid].rename(
                columns={"PID": 'protein', 'sequence': 'seq'})
            node_features = embedder(model, tokenizer, node_df)
            graph = self.create_graph(
                node_df, node_map, node_features, is_orphan=True)
            utils.insert_data(env, orphan_node_pid.encode(),
                              pickle.dumps(graph))

        env.close()

    def _truncate_sequence(self, sequence, target_length=1024):
        
        if len(sequence) > target_length:
            half_length = target_length // 2
            middle_index = len(sequence) // 2
            start_index = max(middle_index - half_length, 0)
            end_index = min(middle_index + half_length, len(sequence))
            return sequence[start_index:end_index]
        else:
            return sequence
    
    def _load_model_and_tokenizer(self):
        
        from transformers import T5Tokenizer
        from transformers import T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained(
            'Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)
        model.full() if self.device == 'cpu' else model.half()

        return model, tokenizer
       

    def _ProstT5_emb_fetch(self, model, tokenizer, flat_ppi_df):
        
        flat_ppi_df["seq"] = flat_ppi_df['seq'].apply(self._truncate_sequence)
        sequences = flat_ppi_df.seq.tolist()
        pids = flat_ppi_df.protein.tolist()
        emb_dict = {}
        for pid, sequence in zip(pids, sequences):
            sequence_examples = [
                " ".join(list(re.sub(r"[UZOB]", "X", sequence)))]
            sequence_examples = ["<AA2fold>" +
                                 " " + s for s in sequence_examples]

            ids = tokenizer.batch_encode_plus(
                sequence_examples,  add_special_tokens=True, padding="longest", return_tensors='pt').to(self.device)
            model.eval()
            with torch.no_grad():
                try:
                    embedding_rpr = model(
                        ids.input_ids, attention_mask=ids.attention_mask)
                except:
                    breakpoint()
            p3DI_emb = embedding_rpr.last_hidden_state[:, 1:-1].mean(dim=1)
            p3DI_emb = p3DI_emb.detach().cpu()
            emb_dict[pid] = p3DI_emb
            del p3DI_emb, ids, embedding_rpr

        return emb_dict
    
