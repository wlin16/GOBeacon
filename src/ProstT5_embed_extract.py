import src.utils as utils
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.lmdb_loader import LMDBLoader

import re
import logging
from tqdm import tqdm
import pickle
import os


class EmbedGenerator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
    
    def _load_model_and_tokenizer(self, model_name):
        if model_name == "ProstT5":
            from transformers import T5Tokenizer
            from transformers import T5EncoderModel
            tokenizer = T5Tokenizer.from_pretrained(
                'Rostlab/ProstT5', do_lower_case=False)
            model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)
            model.full() if self.device == 'cpu' else model.half()
        elif model_name == "ESM2":
            from transformers import AutoTokenizer
            from transformers import EsmModel
            tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
            model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.device)
        else:
            raise ValueError(f"Model name {model_name} is not supported.")
        return model, tokenizer

    def embedding_generate(self, centre_node_csv):
        centre_df = pd.read_pickle(centre_node_csv)
        if "all" in centre_df.index:
            centre_df = centre_df.drop(index = "all")
        centre_df.index.name = 'PID'
        centre_df.reset_index(inplace=True)

        logging.info(f"Generating embeds for dataframe: [{centre_node_csv}]")

        model, tokenizer = self._load_model_and_tokenizer(self.cfg.model_name)
        embedder = {
                    "ProstT5": self._ProstT5_emb_fetch,
                    "ESM2": self._ESM_emb_fetch,
                    }[self.cfg.model_name]
     
        lmdb_path = self.cfg.lmdb_path
        os.makedirs(lmdb_path, exist_ok=True)
        env = utils.initialize(lmdb_path)

        center_node_list = centre_df.PID.tolist()
        centre_df["sequence"] = centre_df['sequence'].apply(
            self._truncate_sequence)

        logging.info(
            f'Generating embeds for {len(center_node_list)} proteins...')

        for index, row in tqdm(centre_df.iterrows(), total=len(centre_df)):
            PID = row['PID']
            seq = row['sequence']
            
            pid_features = embedder(model, tokenizer, seq)

            utils.insert_data(env, PID.encode(),
                              pickle.dumps(pid_features))

        env.close()

    def _truncate_sequence(self, sequence, target_length=999):

        if len(sequence) > target_length:
            half_length = target_length // 2
            middle_index = len(sequence) // 2
            start_index = max(middle_index - half_length, 0)
            end_index = min(middle_index + half_length, len(sequence))
            return sequence[start_index:end_index]
        else:
            return sequence

    def _ProstT5_emb_fetch(self, model, tokenizer, seq):

        sequence_examples = [
            " ".join(list(re.sub(r"[UZOB]", "X", seq)))]
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
        p3DI_emb = embedding_rpr.last_hidden_state[0, 1:-1, :]
        p3DI_emb = p3DI_emb.detach().cpu()
        return p3DI_emb

    def _ESM_emb_fetch(self, model, tokenizer, seq):
        sequence = [seq] #packages sequence in to a list for batch_encode_plus to read

        ids = tokenizer.batch_encode_plus(sequence, return_tensors='pt').to(self.device)
        model.eval()
        with torch.no_grad():
            try:
                embedding_rpr = model(
                    ids.input_ids, attention_mask=ids.attention_mask)
            except:
                breakpoint()
        esm_emb = embedding_rpr.last_hidden_state[0, 1:-1, :]
        esm_emb = esm_emb.detach().cpu()

        return esm_emb
    