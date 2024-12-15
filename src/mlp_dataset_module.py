#/$$
# $ @Author       : Weining Lin
# $ @Date         : 2024-04-14 19:54
# $ @LastEditTime : 2024-05-18 12:39
#$/
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.mlp_dataset import MLP_Dataset
import src.utils as utils


class MLPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
       
        if stage == "fit" or stage is None:
            X_train, y_train = utils.read_df(self.cfg, pid_list=self.cfg.train_sub_list)
            X_valid, y_valid = utils.read_df(self.cfg, pid_list=self.cfg.valid_sub_list)

            self.train_dataset = MLP_Dataset(X_train, y_train, self.cfg)
            self.valid_dataset = MLP_Dataset(X_valid, y_valid, self.cfg)

        if stage == "test" or stage is None:
            X_test, y_test = utils.read_df(self.cfg, mode="test")
            self.test_dataset = MLP_Dataset(X_test, y_test, self.cfg)

        if stage == "predict" or stage is None:
            X_test, y_test = utils.read_df(self.cfg, mode="target")
            self.test_dataset = MLP_Dataset(X_test, y_test, self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=10,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10,
            persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10,
            persistent_workers=True)
