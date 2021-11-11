import pytorch_lightning as pl
from torch import optim
import wandb
import torch
import os

from torch.nn.functional import binary_cross_entropy

import torch_geometric as tg
import torchmetrics
from pytorch_lightning.loggers.wandb import WandbLogger

from GraphCoAttention.datasets.HeterogenousDDI import HeteroDrugDrugInteractionData
# from GraphCoAttention.nn.models.CoAttention import CoAttention
from GraphCoAttention.nn.models.HeterogenousCoAttention import HeteroGNN
# from GraphCoAttention.nn.conv.GATConv import GATConv


class Learner(pl.LightningModule):
    def __init__(self, root_dir, hidden_dim=25, n_cycles=16, n_head=1, dropout=0.1, lr=0.001, bs=2):
        super().__init__()
        self.root_dir = root_dir

        self.dataset = HeteroDrugDrugInteractionData(root=self.root_dir)
        self.dataset = self.dataset.shuffle()

        self.dataset = self.dataset[:10]

        self.num_node_types = len(self.dataset[0].x_dict)
        self.num_workers = 1
        self.n_cycles = n_cycles
        self.n_head = n_head
        self.dropout = dropout
        self.batch_size = bs
        self.lr = lr

        # self.num_features = self.dataset.num_features
        self.hidden_dim = hidden_dim

        wandb.config.hidden_dim = self.hidden_dim
        wandb.config.n_layers = self.n_cycles
        wandb.config.n_head = self.n_head
        wandb.config.dropout = self.dropout

        # self.encoder = GATConv(self.num_features, self.hidden_dim, heads=self.n_head, dropout=self.dropout)
        #
        # self.inner = GATConv(self.hidden_dim * self.n_head, self.hidden_dim, heads=self.n_head,
        #                      add_self_loops=True, bipartite=False, dropout=self.dropout)
        # self.outer = GATConv(self.hidden_dim * self.n_head, self.hidden_dim, heads=self.n_head, add_self_loops=True,
        #                      concat=False, bipartite=True, dropout=self.dropout)
        #
        # self.update = tg.nn.dense.Linear(self.hidden_dim * self.n_head + self.hidden_dim, self.hidden_dim * self.n_head)
        # # self.update = GATConv(self.hidden_dim*self.n_head+self.hidden_dim, self.hidden_dim, heads=self.n_head,
        # #                       add_self_loops=True, bipartite=False, dropout=self.dropout)
        #
        # self.readout = tg.nn.dense.Linear(in_channels=2 * self.hidden_dim, out_channels=1)
        # # self.readout = GATConv(self.hidden_dim*self.n_head, self.hidden_dim, heads=1,
        # #                        add_self_loops=True, bipartite=False, dropout=self.dropout)

        self.HeterogenousCoAttention = HeteroGNN(hidden_channels=self.hidden_dim, out_channels=1, num_layers=self.n_cycles,
                                                 batch_size=self.batch_size, num_node_types=self.num_node_types)

        # self.CoAttention = CoAttention(hidden_channels=self.hidden_dim, encoder=self.encoder,
        #                                outer=self.outer, inner=self.inner,
        #                                update=self.update, readout=self.readout,
        #                                n_cycles=self.n_cycles, batch_size=self.batch_size, n_head=self.n_head)

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch, *args, **kwargs):

        logits = self.HeterogenousCoAttention(batch.x_dict, batch.edge_index_dict, batch)

        # print(logits)
        # exit()
        # logits = self.CoAttention(data)
        # logits = torch.sigmoid(torch.mean(logits))
        return logits

    def training_step(self, data, batch_idx):
        logits = self(data)
        y_pred = logits.squeeze()
        y_true = data.binary_y.float()

        bce = self.bce_loss(input=y_pred, target=y_true)
        # self.log('train_loss', bce)
        wandb.log({"train/loss": bce})
        wandb.log({'train/y_pred': y_pred})
        wandb.log({'train/y_true': y_true})
        return {'loss': bce}  # , 'train_accuracy': acc, 'train_f1': f1}

    def validation_step(self, val_batch, batch_idx):

        # print(val_batch.binary_y.float())

        logits = self(val_batch)
        y_pred = logits.squeeze()
        y_true = val_batch.binary_y.float()

        bce_loss = self.bce_loss(input=y_pred, target=y_true)
        # self.log('validation_loss', bce_loss)
        # self.log('Predicted', y_pred)
        # self.log('Actual', y_true)
        wandb.log({"val/loss": bce_loss})
        return {'loss': bce_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, '25,35', gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return tg.loader.DataLoader(list(self.dataset), batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return tg.loader.DataLoader(list(self.dataset), batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=False, shuffle=True)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    data_dir = os.path.join('GraphCoAttention', 'data')
    wandb_logger = WandbLogger(project='flux', log_model='all')
    trainer = pl.Trainer(gpus=[0], max_epochs=2000, check_val_every_n_epoch=500, accumulate_grad_batches=1)
    trainer.fit(Learner(data_dir, bs=10, lr=0.0005, n_cycles=40, hidden_dim=4))
