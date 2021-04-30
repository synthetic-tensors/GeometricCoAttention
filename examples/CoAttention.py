import pytorch_lightning as pl
import torch
import os

from GraphCoAttention.datasets.DrugInteractionData import DrugDrugInteractionData
from GraphCoAttention.nn.models.CoAttention import CoAttention
from GraphCoAttention.nn.conv.GATConv import GATConv


class Learner(pl.LightningModule):
    def __init__(self, root_dir, hidden_dim=48, n_head=1, dropout=0.1):
        super().__init__()
        self.root_dir = root_dir
        self.dataset = DrugDrugInteractionData(root=self.root_dir)

        self.num_features = self.dataset.num_features
        self.hidden_dim = hidden_dim

        self.outer = GATConv(self.num_features, self.hidden_dim, heads=n_head, add_self_loops=False,
                               concat=False, bipartite=True, dropout=dropout)
        self.inner = GATConv(self.num_features, self.hidden_dim, heads=n_head, dropout=dropout)

        self.update = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.readout = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.CoAttention = CoAttention(hidden_channels=self.hidden_dim, outer=self.outer, inner=self.inner,
                                       update=self.update, readout=self.readout)

    def forward(self, data, *args, **kwargs):
        logits = self.CoAttention(data)
        return logits


if __name__ == '__main__':
    data_dir = os.path.join('GraphCoAttention', 'data')
    trainer = pl.Trainer(gpus=1, max_epochs=20, check_val_every_n_epoch=10, accumulate_grad_batches=25)
    trainer.fit(Learner(data_dir))
