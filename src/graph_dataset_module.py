#/$$
# $ @Author       : Weining Lin
# $ @Date         : 2024-04-14 19:54
# $ @LastEditTime : 2024-04-30 22:27
# $ @Description  : 请填写简介
#$/
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from src.graph_dataset import Graph_Dataset
import src.utils as utils
import logging 

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        
        if stage == "fit" or stage is None:
            X_train = utils.read_txt(self.cfg, self.cfg.train_sub_list)
            X_valid = utils.read_txt(self.cfg, self.cfg.valid_sub_list)

            logging.info(f'X_train shape: {len(X_train)}')
            logging.info(f'X_valid shape: {len(X_valid)}\n')

            self.train_dataset = Graph_Dataset(self.cfg, mode="train", key_list=X_train)
            self.valid_dataset = Graph_Dataset(self.cfg, mode="valid", key_list=X_valid)

        if stage == "test" or stage is None:
            self.test_dataset = Graph_Dataset(self.cfg, mode="test")

        if stage == "predict" or stage is None:
            self.test_dataset = Graph_Dataset(self.cfg, mode="infer")

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=20,
                persistent_workers=True, pin_memory=True, follow_batch=['x', 'positive_x', 'negative_x', 'para_x'],
                )
        # return DataLoader(
        #         self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, follow_batch=['x', 'positive_x', 'negative_x', 'para_x'],
        #         )


    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=20,
                persistent_workers=True, pin_memory=True, follow_batch=['x', 'positive_x', 'negative_x', 'para_x'],
                )
        # return DataLoader(
        #         self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=False, follow_batch=['x', 'positive_x', 'negative_x', 'para_x'],
        #         )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=20,
                persistent_workers=True, pin_memory=True, follow_batch=['x', 'positive_x', 'negative_x', 'para_x'])
        # return DataLoader(
        #         self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False,)

    def predict_dataloader(self):
        return DataLoader(
                self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=20,
                persistent_workers=True, pin_memory=True, follow_batch=['x', 'positive_x', 'negative_x', 'para_x'],
                )
        # return DataLoader(
        #         self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False,
        #         )
