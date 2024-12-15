import torch
from torch import nn
import numpy as np
import random
from torch_geometric.data import Data

class HetData(Data):
    def __init__(self, x, edge_index, edge_attr,
                 positive_x=None, positive_edge_index=None, positive_edge_attr=None,
                 para_x=None, para_edge_index=None, para_edge_attr=None,
                 negative_x=None, negative_edge_index=None, negative_edge_attr=None,
                 ssl1_mask=None, ssl2_mask=None, anchor_label=None, y_true=None, pid=None, mode = None):
        super(HetData, self).__init__()

        self.x, self.edge_index, self.edge_attr = x, edge_index, edge_attr
        self.positive_x, self.positive_edge_index, self.positive_edge_attr = positive_x, positive_edge_index, positive_edge_attr
        self.para_x, self.para_edge_index, self.para_edge_attr = para_x, para_edge_index, para_edge_attr
        self.negative_x, self.negative_edge_index, self.negative_edge_attr = negative_x, negative_edge_index, negative_edge_attr
        self.ssl1_mask, self.ssl2_mask, self.anchor_label = ssl1_mask, ssl2_mask, anchor_label
        self.y_true = y_true
        self.pid = pid
        #self.label, self.contrast_mask = label, contrast_mask

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'y_true':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
    
    def __inc__(self, key, value,  *args, **kwargs):
        if 'edge_index' in key:
            return getattr(self, key[:-10]+"x").size(0)
        else:
            return super().__inc__(key, value)
