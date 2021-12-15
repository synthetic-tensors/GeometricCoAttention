import pytorch_lightning as pl
from torch import optim
import wandb
import torch
import os

from torch.nn.functional import binary_cross_entropy

import torch_geometric as tg
import torchmetrics
from pytorch_lightning.loggers.wandb import WandbLogger

from GraphCoAttention.nn.models.HeterogenousCoAttention import HeteroGNN
#import dataloader


class Learner(pl.LightningModule):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

        self.dataset = HeteroDrugDrugInteractionData(root=self.root_dir) #change to appropriate data loader
        self.dataset = self.dataset.shuffle()

        self.dataset = self.dataset[:10]

        self.num_node_types = len(self.dataset[0].x_dict)
        self.num_workers = 32
        self.n_cycles = config.n_cycles
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.hidden_dim = config.hidden_dim

        self.HeterogenousCoAttention = HeteroGNN(hidden_channels=self.hidden_dim, out_channels=1, num_layers=self.n_cycles,
                                                 batch_size=self.batch_size, num_node_types=self.num_node_types,
                                                 num_heads=self.n_head)

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
        metrics = {"train/loss": bce,"train/y_pred": wandb.Histogram(y_pred.cpu().detach()), "train/y_true": wandb.Histogram(y_true.cpu().detach()), "epoch": self.current_epoch}
        wandb.log(metrics)
        # wandb.log({"train/loss": bce})
        # wandb.log({"train/y_pred": wandb.Histogram(y_pred)})
        # wandb.log({"train/y_true": wandb.Histogram(y_true)})
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
        metrics = {"val/loss": bce_loss,"val/y_pred": wandb.Histogram(y_pred.cpu().detach()), "val/y_true": wandb.Histogram(y_true.cpu().detach())}
        wandb.log(metrics)
        # wandb.log({"val/y_pred": wandb.Histogram(y_pred)})
        # wandb.log({"val/y_true": wandb.Histogram(y_true)})
        # wandb.log({"val/loss": bce_loss})
        return {'loss': bce_loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.28, 0.93), weight_decay=0.01)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, '25,35', gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return tg.loader.DataLoader(list(self.dataset), batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return tg.loader.DataLoader(list(self.dataset), batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=False, shuffle=True)


    
hyperparameter_defaults = dict(
    learning_rate = 0.001,
    batch_size=2,
    n_head=1,
    hidden_dim=25,
    n_cycles=16,
    dropout=0.1
    )

wandb.init(config=hyperparameter_defaults)
config = wandb.config

# wandb.agent("dev-eloper", project="flux", function=train)
wandb_logger = WandbLogger(project='flux', log_model='all')

data_dir = os.path.join('GraphCoAttention', 'data')
trainer = pl.Trainer(gpus=[0], max_epochs=500, check_val_every_n_epoch=100, accumulate_grad_batches=1)
trainer.fit(Learner(data_dir))
