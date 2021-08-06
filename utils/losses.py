'''
An aggregation of special loss functions.
'''
import torch
import torch.nn as nn
import torch.distributed as dist


class NT_Xent(nn.Module):
    '''
    NT_Xent loss implementation based on SimCLR implementation:
    https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py

    Some adaptations have been made, e.g. as we do not do distributed training
    w.r.t. contrastive learning all the `world_size` parts are gone.
    Furthermore, the pairs are coming from one and the same batch, not two
    separate ones, so the input handling/preprocessing is a little bit
    different.
    '''
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    @classmethod
    def mask_correlated_samples(cls, y):
        return ~(y.reshape(1,-1) == y.reshape(-1,1))

    def forward(self, x, y):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = x.shape[0]

        sim = self.similarity_f(x.unsqueeze(1), x.unsqueeze(0)) / self.temperature

        # We have N samples
        mask = NT_Xent.mask_correlated_samples(y)
        positive_samples = sim[~mask].reshape(N, -1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
