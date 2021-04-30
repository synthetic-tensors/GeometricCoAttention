import torch
from torch.nn import Parameter

from GraphCoAttention.data.MultipartiteData import BipartitePairData


class CoAttention(torch.nn.Module):
    r"""The Graph Co-Attention model from the
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

    def __init__(self, hidden_channels, inner, outer, update, readout):
        super(CoAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.inner_i, self.inner_j = inner, inner
        self.outer_i, self.outer_j = outer, outer
        self.update_i, self.update_j = update, update
        self.readout = readout

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

    def forward(self, data: BipartitePairData, *args, **kwargs):

        m_i = self.inner_i(x=data.x_i, edge_index=data.inner_edge_index_i)
        m_j = self.inner_j(x=data.x_j, edge_index=data.inner_edge_index_j)
        a_ij = self.outer_i(x=(data.x_j, data.x_i), edge_index=data.outer_edge_index_j)
        a_ji = self.outer_j(x=(data.x_i, data.x_j), edge_index=data.outer_edge_index_i)

        i = self.update_i(m_i, a_ij)
        j = self.update_j(m_j, a_ji)

        logits = self.readout(i, j)

        return logits