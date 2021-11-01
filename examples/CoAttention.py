import pytorch_lightning as pl
from torch import optim
import torch
import os

from torch.nn.functional import binary_cross_entropy

import torch_geometric as tg

from GraphCoAttention.datasets.DrugInteractionData import DrugDrugInteractionData
from GraphCoAttention.nn.models.CoAttention import CoAttention
from GraphCoAttention.nn.conv.GATConv import GATConv


class Learner(pl.LightningModule):
    def __init__(self, root_dir, hidden_dim=48, n_head=1, dropout=0.1):
        super().__init__()
        self.root_dir = root_dir
        self.dataset = DrugDrugInteractionData(root=self.root_dir)
        self.num_workers = 1

        self.num_features = self.dataset.num_features

        self.hidden_dim = hidden_dim

        self.outer = GATConv(self.hidden_dim, self.hidden_dim, heads=n_head, add_self_loops=False,
                             concat=False, bipartite=True, dropout=dropout)

        self.encoder = GATConv(self.num_features, self.hidden_dim, heads=n_head, dropout=dropout)
        
        self.inner = GATConv(self.hidden_dim, self.hidden_dim, heads=n_head, dropout=dropout)

        self.update = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.readout = torch.nn.Linear(self.hidden_dim, 1)

        self.CoAttention = CoAttention(hidden_channels=self.hidden_dim, encoder=self.encoder,
                                       outer=self.outer, inner=self.inner,
                                       update=self.update, readout=self.readout,
                                       n_cycles=4)

    def forward(self, data, *args, **kwargs):
        logits = self.CoAttention(data)
        logits = torch.tanh(torch.mean(logits))
        return logits

    def training_step(self, data, batch_idx):
        logits = self(data)
        loss = binary_cross_entropy(input=logits.unsqueeze(0), target=data.binary_y.float())
        return {'loss': loss}  # , 'train_accuracy': acc, 'train_f1': f1}

    def validation_step(self, val_batch, batch_idx):
        logits = self(val_batch)
        loss = binary_cross_entropy(input=logits.unsqueeze(0), target=val_batch.binary_y.float())
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005, weight_decay=1e-4, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, '25,35', gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return tg.data.DataLoader(self.dataset, batch_size=1, follow_batch=['x_i', 'x_j'],
                                  num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return tg.data.DataLoader(self.dataset, batch_size=1, follow_batch=['x_i', 'x_j'],
                                  num_workers=self.num_workers, pin_memory=True, shuffle=True)


if __name__ == '__main__':
    data_dir = os.path.join('GraphCoAttention', 'data')
    trainer = pl.Trainer(gpus=[1], max_epochs=20, check_val_every_n_epoch=10, accumulate_grad_batches=5)
    trainer.fit(Learner(data_dir))
