from src.triplet_loss import TripletLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.nn import GINConv, global_add_pool
from src.nt_xent import NT_Xent
import torch
import math
from typing import List, Tuple, Optional



class CustomMLP(nn.Module):
    def __init__(self, config):
        super(CustomMLP, self).__init__()

        num_input = config.num_features
        num_hidden = config.hidden
        num_output = config.label_num

        self.config = config
        self.hidden = nn.Linear(num_input, num_hidden)
        self.predict = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.SiLU(inplace=True),
            nn.Linear(num_hidden, int(num_hidden/2)),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(int(num_hidden/2), num_output)

        self.label_emb = nn.Sequential(
        nn.Linear(num_output, num_output),
        nn.SiLU(inplace=True),
        nn.Linear(num_output, num_output)
        )
        self.info_nce = NT_Xent
    
    def mask_normalise(self, x, mask):
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = x * mask_expanded
        sum_mask = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)
        cls = x.sum(dim=1, keepdim=True) / sum_mask
        return cls.squeeze(1)

    def forward(self,x, mask, labels, SSL=True, mode="train"):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.hidden(x)
        x = self.predict(x)

        cls = self.fc(x)
        cls = self.mask_normalise(cls, mask)
        if SSL:
            if mode == "train":
                assert torch.is_tensor(labels) == True
                ssl_loss = self.label_contrastive(cls, labels)
                return cls, ssl_loss
            else:
                criterion = nn.BCEWithLogitsLoss(reduction='mean')
                return cls, criterion(cls, labels)
        else:
            return cls
        
    def label_contrastive(self, x_feat, labels):

        y_label_emb = self.label_emb(labels)
        
        criterion = self.info_nce(labels.shape[0], 0.1, 1)
        y_true_mask = (torch.cat([labels, labels]).sum(-1) > 0.5).float()
        ssl_loss = criterion(x_feat, y_label_emb) * y_true_mask
        ssl_loss = 0.05 * ssl_loss.sum() / y_true_mask.sum()

        return ssl_loss


class GoModel(nn.Module):
    def __init__(self, cfg):
        super(GoModel, self).__init__()

        self.gnn_encoder = GATNet(cfg)

        self.readout = nn.Sequential(
                        nn.Linear(256,256),
                        nn.ReLU(),
                        nn.Linear(256, cfg.label_num))
        
        self.label_emb = nn.Sequential(
                        nn.Linear(cfg.label_num, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256))
        
        self.ssl_loss = TripletLoss(margin=2.0)
        self.info_nce = NT_Xent
        self.cfg = cfg

    def forward(self, data, SSL=True, mode="train"):
        data.x = torch.tensor(data.x, dtype=torch.float32)
        anchor_x = self.gnn_encoder(data.x, data.edge_index, data.edge_attr, data.x_batch)
        y_pred_anchor = self.readout(anchor_x)

        if SSL:
            if mode == "train":
                ssl_loss = self.label_contrastive(data, anchor_x)
                
                return y_pred_anchor, torch.mean(ssl_loss)
            else:
                criterion = nn.BCEWithLogitsLoss(reduction='mean')
                return y_pred_anchor, criterion(y_pred_anchor, data.y_true)
        else:
            return y_pred_anchor

    
    def label_contrastive(self, data, x_feat):

        y_label_emb = self.label_emb(data.y_true)
        
        criterion = self.info_nce(data.y_true.shape[0], 0.1, 1)
        y_true_mask = (torch.cat([data.y_true, data.y_true]).sum(-1) > 0.5).float()
        ssl_loss = criterion(x_feat, y_label_emb) * y_true_mask
        ssl_loss = 0.05 * ssl_loss.sum() / y_true_mask.sum()

        return ssl_loss

class GATNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for i in range(cfg.num_layers):
            conv = GATConv(cfg.num_features if i == 0 else cfg.hidden *
                           cfg.num_heads, cfg.hidden, heads=cfg.num_heads)
            self.convs.append(conv)
            self.layer_norms.append(LayerNorm(cfg.hidden * cfg.num_heads))

        self.lin1 = nn.Linear(cfg.hidden * cfg.num_heads, cfg.hidden)
        self.layer_norm1 = LayerNorm(cfg.hidden)
        self.lin2 = nn.Linear(cfg.hidden, cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, edge_index, edge_attr, batch):

        x = torch.tensor(x, dtype=torch.float32).squeeze(1)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        for conv, layer_norm in zip(self.convs, self.layer_norms):
            x = F.relu(layer_norm(conv(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.layer_norm1(self.lin1(x)))
        x = self.dropout(x)
        x = self.lin2(x)

        return x