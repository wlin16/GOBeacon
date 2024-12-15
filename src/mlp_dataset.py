#/$$
# $ @Author       : Weining Lin
# $ @Date         : 2024-04-14 18:51
# $ @LastEditTime : 2024-04-14 20:17
#$/


import torch
from torch.utils.data import Dataset
from src.lmdb_loader import LMDBLoader

class MLP_Dataset(Dataset):
    def __init__(self, key_list, label_list, cfg, max_tensor_length=1000):
        super().__init__()
        self.key_list = key_list
        self.label_list = label_list
        self.label_num = cfg[f'{cfg.sub_ontology}_len']
        self.max_tensor_length = max_tensor_length
        self.lmdb_dataset = LMDBLoader(cfg.lmdb_path, key_list)

    def __len__(self):
        return len(self.key_list)

    def ohe_transform(self, label_index):
        label = torch.zeros(self.label_num)
        label[list(label_index)] = 1
        return label

    def pad_tensor(self, tensor):
        seq_len, feature_dim = tensor.shape
        if seq_len < self.max_tensor_length:
            
            padding_size = self.max_tensor_length - seq_len
            
            padding = torch.zeros(
                (padding_size, feature_dim), dtype=tensor.dtype)
            padded_tensor = torch.cat([tensor, padding], dim=0)
            
            mask = torch.cat([torch.ones(seq_len, dtype=torch.long),
                              torch.zeros(padding_size, dtype=torch.long)], dim=0)
        else:
            padded_tensor = tensor[:self.max_tensor_length]
            mask = torch.ones(self.max_tensor_length, dtype=torch.long)

        return padded_tensor, mask

    def __getitem__(self, idx):
        data = self.lmdb_dataset[idx]
        padded_data, mask = self.pad_tensor(data)
        label = self.label_list[idx]
        label = self.ohe_transform(label)

        return padded_data, mask, label

