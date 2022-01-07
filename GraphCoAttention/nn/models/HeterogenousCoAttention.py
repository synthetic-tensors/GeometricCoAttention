import torch
from torch import nn
from torch.nn import Parameter, Sequential, ReLU, GRU
from torch.nn import functional as F
import torch_geometric as tg

from torch_geometric.nn import GATConv, HeteroConv, Linear, GATv2Conv, NNConv, Set2Set
from torch_geometric.nn.glob import global_mean_pool, global_add_pool
from torch.nn import LeakyReLU

from GraphCoAttention.data.MultipartiteData import BipartitePairData


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, outer_out_channels, inner_out_channels,
                 num_layers, batch_size, num_node_types, num_heads):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_channels = hidden_channels
        self.heads = num_heads

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('x_i', 'inner_edge_i', 'x_i'): GATv2Conv(-1, self.hidden_channels, heads=num_heads),
                ('x_j', 'inner_edge_j', 'x_j'): GATv2Conv(-1, self.hidden_channels, heads=num_heads),
                ('x_i', 'outer_edge_ij', 'x_j'): GATv2Conv(-1, self.hidden_channels, heads=num_heads),
                ('x_j', 'outer_edge_ji', 'x_i'): GATv2Conv(-1, self.hidden_channels, heads=num_heads),
                ('x_i', 'inner_edge_i', 'x_i'): GATv2Conv(-1, self.hidden_channels, heads=num_heads),
                ('x_j', 'inner_edge_j', 'x_j'): GATv2Conv(-1, self.hidden_channels, heads=num_heads),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(self.hidden_channels, outer_out_channels)

        self.lin_i = Linear(self.hidden_channels, inner_out_channels)
        self.lin_j = Linear(self.hidden_channels, inner_out_channels)
        # self.hlin = tg.nn.HeteroLinear(hidden_channels, out_channels, num_node_types=num_node_types)

    def forward(self, x_dict, edge_index_dict, d):

        # x_dict, edge_index_dict = x_dict, edge_index_dict

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: torch.tanh(torch.sum(x.view(-1, self.heads, self.hidden_channels), dim=1))
                      for key, x in x_dict.items()}

            # [print(key, x.shape) for key, x in x_dict.items()]
            # [print(key, x.view(-1, self.heads, self.hidden_channels).shape) for key, x in x_dict.items()]
            # [print(key, torch.mean(x.view(-1, self.heads, self.hidden_channels), dim=1).shape) for key, x in x_dict.items()]

        # p_i = F.leaky_relu(global_add_pool(x_dict['x_i'], batch=d['x_i'].batch, size=self.batch_size).unsqueeze(1))
        # p_j = F.leaky_relu(global_add_pool(x_dict['x_j'], batch=d['x_j'].batch, size=self.batch_size).unsqueeze(1))

        # p_i = global_add_pool(x_dict['x_i'], batch=d['x_i'].batch, size=self.batch_size).unsqueeze(1).sigmoid()
        # p_j = global_add_pool(x_dict['x_j'], batch=d['x_j'].batch, size=self.batch_size).unsqueeze(1).sigmoid()

        p_i = global_add_pool(x_dict['x_i'], batch=d['x_i'].batch, size=self.batch_size).unsqueeze(1).tanh()
        p_j = global_add_pool(x_dict['x_j'], batch=d['x_j'].batch, size=self.batch_size).unsqueeze(1).tanh()

        y_i_ = self.lin_i(p_i)
        y_j_ = self.lin_j(p_j)

        x = torch.cat([p_i, p_j], dim=1)
        x = torch.sum(x, dim=1)

        logits = self.lin(x).sigmoid()
        return logits, y_i_, y_j_


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

            m_i = F.leaky_relu(m_i)
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

        logits = self.readout(x)
        logits = torch.sigmoid(torch.mean(logits, dim=1))
        # logits = torch.mean(logits, dim=1)

        return logits
    
    
class Net(torch.nn.Module):
    def __init__(self, hidden_channels, outer_out_channels, inner_out_channels,
                 num_layers, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.dim = hidden_channels

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, self.dim * self.dim))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('x_i', 'inner_edge_i', 'x_i'): NNConv(self.dim, self.dim, nn, aggr='mean'),
                ('x_j', 'inner_edge_j', 'x_j'): NNConv(self.dim, self.dim, nn, aggr='mean'),
                ('x_i', 'outer_edge_ij', 'x_j'): NNConv(self.dim, self.dim, nn, aggr='mean'),
                ('x_j', 'outer_edge_ji', 'x_i'): NNConv(self.dim, self.dim, nn, aggr='mean'),
                ('x_i', 'inner_edge_i', 'x_i'): NNConv(self.dim, self.dim, nn, aggr='mean'),
                ('x_j', 'inner_edge_j', 'x_j'): NNConv(self.dim, self.dim, nn, aggr='mean'),
            }, aggr='sum')
            self.convs.append(conv)
        
        self.lin = Linear(self.dim, outer_out_channels)

        self.lin_i = Linear(self.dim, inner_out_channels)
        self.lin_j = Linear(self.dim, inner_out_channels)
        

    def forward(self, x_dict, edge_index_dict, d):

        x_dict, edge_index_dict = x_dict, edge_index_dict
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: torch.tanh(torch.sum(x.view(-1, self.dim), dim=1))
                      for key, x in x_dict.items()}

        p_i = global_add_pool(x_dict['x_i'], batch=d['x_i'].batch, size=self.batch_size).unsqueeze(1).tanh()
        p_j = global_add_pool(x_dict['x_j'], batch=d['x_j'].batch, size=self.batch_size).unsqueeze(1).tanh()
        
        y_i_ = self.lin_i(p_i)
        y_j_ = self.lin_j(p_j)

        x = torch.cat([p_i, p_j], dim=1)
        x = torch.sum(x, dim=1)

        logits = self.lin(x).sigmoid()
        return logits, y_i_, y_j_
    
    

#     def __init__(self, dataset, dim):
#         super().__init__()

#         self.num_features = len(dataset[0].x_dict)

#         self.lin0 = Linear(self.num_features, dim)

#         nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
#         self.conv = NNConv(dim, dim, nn, aggr='mean')
#         self.gru = GRU(dim, dim)

#         self.set2set = Set2Set(dim, processing_steps=3)
        
#         self.lin1 = Linear(2 * dim, dim)
#         self.lin2 = Linear(dim, 1)
        
#     def forward(self, x_dict, edge_index_dict, d):
#         out = F.relu(self.lin0(d['x_i'].x)) #check this
#         h = out.unsqueeze(0)

#         for i in range(3):
#             m = F.relu(self.conv(out, d.edge_index, d.edge_attr)) #check this
#             out, h = self.gru(m.unsqueeze(0), h)
#             out = out.squeeze(0)

#         out = self.set2set(out, d)
        
#         y_i_ = self.lin1(out)
#         y_j_ = self.lin2(out)

#         logits = self.lin(out).sigmoid()
#         return logits, y_i_, y_j_
