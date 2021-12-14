import pytorch_lightning as pl
from torch import optim
import wandb
import torch
import os

from torch.nn.functional import binary_cross_entropy

import torch_geometric as tg
import torchmetrics
from pytorch_lightning.loggers.wandb import WandbLogger

from GraphCoAttention.datasets.HeterogenousDDI import HeteroDrugDrugInteractionData, HeteroQM9
# from GraphCoAttention.nn.models.CoAttention import CoAttention
from GraphCoAttention.nn.models.HeterogenousCoAttention import HeteroGNN
# from GraphCoAttention.nn.conv.GATConv import GATConv


class Learner(pl.LightningModule):
    def __init__(self, root_dir, hidden_dim=25, n_cycles=16, n_head=1, dropout=0.1, lr=0.001, bs=2):
        super().__init__()
        self.root_dir = root_dir

        self.ddi_dataset = HeteroDrugDrugInteractionData(root=self.root_dir)  # .shuffle()
        self.qm9_dataset = HeteroQM9(root=self.root_dir)   # [:10]  # .shuffle()

        # self.dataset = self.dataset[:10]

        self.num_node_types = len(self.qm9_dataset[0].x_dict)
        self.num_workers = 32
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

        self.HeterogenousCoAttention = HeteroGNN(hidden_channels=self.hidden_dim, outer_out_channels=1,
                                                 inner_out_channels=15, num_layers=self.n_cycles,
                                                 batch_size=self.batch_size, num_node_types=self.num_node_types,
                                                 num_heads=self.n_head)

        # self.CoAttention = CoAttention(hidden_channels=self.hidden_dim, encoder=self.encoder,
        #                                outer=self.outer, inner=self.inner,
        #                                update=self.update, readout=self.readout,
        #                                n_cycles=self.n_cycles, batch_size=self.batch_size, n_head=self.n_head)

        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.L1Loss()

    def forward(self, batch, *args, **kwargs):

        y_ij, y_i_, y_j_ = self.HeterogenousCoAttention(batch.x_dict, batch.edge_index_dict, batch)

        # logits = self.CoAttention(data)
        # logits = torch.sigmoid(torch.mean(logits))
        return y_ij, y_i_, y_j_

    def training_step(self, data, batch_idx):

        _, y_i_, y_j_ = self(data['QM9'])
        mse1 = self.mse_loss(input=y_i_.flatten(), target=data['QM9']['y_i'].y_norm)
        mse2 = self.mse_loss(input=y_j_.flatten(), target=data['QM9']['y_j'].y_norm)
        mse = mse1 + mse2

        y_ij, _, _ = self(data['DDI'])
        y_pred = y_ij.squeeze()
        y_true = data['DDI'].binary_y.float()
        bce = self.bce_loss(input=y_pred, target=y_true)

        loss = mse + bce

        wandb.log({"train/loss": loss})

        wandb.log({'train/y_i_pred': y_i_.flatten()})
        wandb.log({'train/y_i_true': data['QM9']['y_i'].y_norm})
        wandb.log({'train/y_j_pred': y_j_.flatten()})
        wandb.log({'train/y_j_true': data['QM9']['y_j'].y_norm})

        wandb.log({"train/y_pred": wandb.Histogram(y_pred.cpu().detach())})
        wandb.log({"train/y_true": wandb.Histogram(y_true.cpu().detach())})

        return {'loss': loss}  # , 'train_accuracy': acc, 'train_f1': f1}

    def validation_step(self, val_batch, batch_idx, loader_idx):

        y_ij, y_i_, y_j_ = self(val_batch)

        if loader_idx == 1:
            y_pred = y_ij.squeeze()
            y_true = val_batch.binary_y.float()
            bce = self.bce_loss(input=y_pred, target=y_true)
            wandb.log({"val/loss": bce})
            loss = bce

        if loader_idx == 0:
            mse1 = self.mse_loss(input=y_i_.flatten(), target=val_batch['y_i'].y_norm)
            mse2 = self.mse_loss(input=y_j_.flatten(), target=val_batch['y_j'].y_norm)
            mse = mse1 + mse2
            wandb.log({"val/loss": mse})
            loss = mse

        # print(val_batch, loader_idx, batch_idx)
        # print(ddi_batch, type(ddi_batch))

        # y_ij, y_i_, y_j_ = self(val_batch)
        # y_pred = y_ij.squeeze()
        # y_true = val_batch.binary_y.float()
        #
        # mse1 = self.mse_loss(input=y_i_.flatten(), target=val_batch['y_i'].y_norm)
        # mse2 = self.mse_loss(input=y_j_.flatten(), target=val_batch['y_j'].y_norm)
        # mse = mse1 + mse2
        # bce = self.bce_loss(input=y_pred, target=y_true)
        # loss = mse
        # # self.log('validation_loss', bce_loss)
        # # self.log('Predicted', y_pred)
        # # self.log('Actual', y_true)
        # wandb.log({"val/loss": loss})

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.28, 0.93), weight_decay=0.01)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, '25,35', gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        qm9_dataloader = tg.loader.DataLoader(list(self.qm9_dataset), batch_size=self.batch_size,
                                              num_workers=self.num_workers, pin_memory=False, shuffle=True)

        ddi_dataloader = tg.loader.DataLoader(list(self.ddi_dataset), batch_size=self.batch_size,
                                              num_workers=self.num_workers, pin_memory=False, shuffle=True)

        loaders = {"QM9": qm9_dataloader, 'DDI': ddi_dataloader}
        return loaders

    def val_dataloader(self):
        qm9_dataloader = tg.loader.DataLoader(list(self.qm9_dataset), batch_size=self.batch_size,
                                              num_workers=self.num_workers, pin_memory=False, shuffle=True)

        ddi_dataloader = tg.loader.DataLoader(list(self.ddi_dataset), batch_size=self.batch_size,
                                              num_workers=self.num_workers, pin_memory=False, shuffle=True)
        # loaders = {"QM9": qm9_dataloader, 'DDI': ddi_dataloader}
        loaders = [qm9_dataloader, ddi_dataloader]
        return loaders


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    data_dir = os.path.join('GraphCoAttention', 'data')
    wandb.init()
    wandb_logger = WandbLogger(project='flux', log_model='all')
    trainer = pl.Trainer(gpus=[1], max_epochs=2000, check_val_every_n_epoch=500, accumulate_grad_batches=1)
    trainer.fit(Learner(data_dir, bs=10, lr=0.001, n_cycles=8, hidden_dim=225, n_head=5))

