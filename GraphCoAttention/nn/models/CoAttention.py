import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import torch_geometric as tg
from torch_geometric.nn.glob import global_mean_pool

from GraphCoAttention.data.MultipartiteData import BipartitePairData


class CoAttention(torch.nn.Module):
    r"""
    The Graph Co-Attention model from the
    `"Drug-Drug Adverse Effect Prediction with Graph Co-Attention" <https://arxiv.org/abs/1905.00534>`_
    paper based on user-defined inner message passing network :math:`\mathcal{I}`
    , outer message passing network :math:`\mathcal{O}`, update layer :math:`\mathcal{U}`,
    and readout layers :math:`\mathcal{R}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        inner (Module): The inner message passing module :math:`\mathcal{I}`.
        outer (Module): The outer message passing module :math:`\mathcal{O}`.
        update (Module): The update module :math:`\mathcal{U}`.
        readout (Module): The readout function :math:`\mathcal{R}`.
    """

    def __init__(self, hidden_channels: int, batch_size: int,  encoder: nn.Module,
                 inner: nn.Module, outer: nn.Module,
                 update: nn.Module, readout: nn.Module,
                 n_cycles: int = 2, n_head=1):
        super(CoAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.batch_size = batch_size
        self.n_cycles = n_cycles
        self.n_head = n_head

        # Initial Dimensionality Expansion
        self.encoder_i, self.encoder_j = encoder, encoder
        # Inner Modules * Number of Cycles
        self.inner_i = nn.ModuleList([inner for i in range(n_cycles)])
        self.inner_j = nn.ModuleList([inner for i in range(n_cycles)])
        # Outer Modules * Number of Cycles
        self.outer_i = nn.ModuleList([outer for i in range(n_cycles)])
        self.outer_j = nn.ModuleList([outer for i in range(n_cycles)])
        # Update Function * Number of Cycles
        self.update_i = nn.ModuleList([update for i in range(n_cycles)])
        self.update_j = nn.ModuleList([update for i in range(n_cycles)])
        # Readouts
        # self.readout_i = readout
        # self.readout_j = readout
        self.readout = readout

        # self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

    def forward(self, data: BipartitePairData, *args, **kwargs):

        x_i = self.encoder_i(x=data.x_i.float(), edge_index=data.inner_edge_index_i)
        x_j = self.encoder_j(x=data.x_j.float(), edge_index=data.inner_edge_index_j)

        for index in range(self.n_cycles):
            m_i = self.inner_i[index](x=x_i, edge_index=data.inner_edge_index_i)
            m_j = self.inner_j[index](x=x_j, edge_index=data.inner_edge_index_j)

            m_i = F.tanh(m_i)
            m_j = F.leaky_relu(m_j)

            a_ij = self.outer_i[index](x=(x_j, x_i), edge_index=data.outer_edge_index_j)
            a_ji = self.outer_j[index](x=(x_i, x_j), edge_index=data.outer_edge_index_i)

            a_ij = F.leaky_relu(a_ij)
            a_ji = F.leaky_relu(a_ji)

            u_i = torch.cat((m_i, a_ij), dim=1)
            u_j = torch.cat((m_j, a_ji), dim=1)

            x_i = self.update_i[index](u_i)
            x_j = self.update_j[index](u_j)
            # x_i = self.update_i[index](x=torch.cat((m_i, a_ij), dim=1), edge_index=data.inner_edge_index_i)
            # x_j = self.update_j[index](x=torch.cat((m_j, a_ji), dim=1), edge_index=data.inner_edge_index_j)

            x_i = F.leaky_relu(x_i)
            x_j = F.leaky_relu(x_j)

        p_i = global_mean_pool(x_i, batch=data.x_i_batch, size=self.batch_size)
        p_j = global_mean_pool(x_j, batch=data.x_j_batch, size=self.batch_size)
        x = torch.cat((p_i, p_j), dim=1)
        x = x.view(self.batch_size, self.n_head, -1)

        logits = self.readout(x).tanh()
        logits = torch.sigmoid(torch.mean(logits, dim=1))
        # logits = torch.mean(logits, dim=1)

        return logits
