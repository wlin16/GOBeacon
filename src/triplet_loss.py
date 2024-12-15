#/$$
# $ @Author       : Weining Lin
# $ @Date         : 2024-05-18 12:43
# $ @LastEditTime : 2024-05-18 12:43
#$/

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=0.15, p=1, reduce=False)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1).to(anchor.device)
            ap_dist = torch.norm(anchor-pos, 1, dim=-1).view(-1)
            an_dist = torch.norm(anchor-neg, 1, dim=-1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


if __name__ == '__main__':
    pass