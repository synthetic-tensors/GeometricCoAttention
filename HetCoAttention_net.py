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
from GraphCoAttention.nn.models.HeterogenousCoAttention import Net
# from GraphCoAttention.nn.conv.GATConv import GATConv


class Learner(pl.LightningModule):
    def __init__(self, root_dir, lr=0.001):
        super().__init__()
        self.root_dir = root_dir

        # self.dataset = HeteroDrugDrugInteractionData(root=self.root_dir)
        self.dataset = HeteroQM9(root=self.root_dir)
        self.dataset = self.dataset.shuffle()

        self.num_workers = 32
        self.lr = lr
        self.dim = 64

        self.Net = Net(dataset=self.dataset, dim=self.dim)

        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        
        print(self.dataset[0])
        exit()
    def forward(self, batch, *args, **kwargs):

        y_ij, y_i_, y_j_ = self.Net(batch)

        # logits = self.CoAttention(data)
        # logits = torch.sigmoid(torch.mean(logits))
        return y_ij, y_i_, y_j_

    def training_step(self, data, batch_idx):
        y_ij, y_i_, y_j_ = self(data)
        y_pred = y_ij.squeeze()
        y_true = data.binary_y.float()

        mse1 = self.mse_loss(input=y_i_.flatten(), target=data['y_i'].y)
        mse2 = self.mse_loss(input=y_j_.flatten(), target=data['y_j'].y)
        mse = mse1 + mse2
        bce = self.bce_loss(input=y_pred, target=y_true)
        loss = bce + mse

        # self.log('train_loss', bce)
        wandb.log({"train/loss": loss})
        wandb.log({'train/y_pred': y_pred})
        wandb.log({'train/y_true': y_true})
        return {'loss': loss}  # , 'train_accuracy': acc, 'train_f1': f1}

    def validation_step(self, val_batch, batch_idx):

        # print(val_batch.binary_y.float())

        y_ij, y_i_, y_j_ = self(val_batch)
        y_pred = y_ij.squeeze()
        y_true = val_batch.binary_y.float()

        mse1 = self.mse_loss(input=y_i_.flatten(), target=val_batch['y_i'].y)
        mse2 = self.mse_loss(input=y_j_.flatten(), target=val_batch['y_j'].y)
        mse = mse1 + mse2
        bce = self.bce_loss(input=y_pred, target=y_true)
        loss = bce + mse
        # self.log('validation_loss', bce_loss)
        # self.log('Predicted', y_pred)
        # self.log('Actual', y_true)
        wandb.log({"val/loss": loss})
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.28, 0.93), weight_decay=0.01)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, '25,35', gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return tg.loader.DataLoader(list(self.dataset),
                                    num_workers=self.num_workers, pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return tg.loader.DataLoader(list(self.dataset), 
                                    num_workers=self.num_workers, pin_memory=False, shuffle=True)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    data_dir = os.path.join('GraphCoAttention', 'data')
    wandb.init()
    wandb_logger = WandbLogger(project='flux', log_model='all')
    trainer = pl.Trainer(gpus=[0], max_epochs=2000, check_val_every_n_epoch=500, accumulate_grad_batches=1)
    trainer.fit(Learner(data_dir))
