import torch_geometric as tg
import torch


class BipartitePairData(tg.data.Data):
    def __init__(self,
                 inner_edge_index_i=None, x_i=None, outer_edge_index_i=None,
                 inner_edge_index_j=None, x_j=None, outer_edge_index_j=None, **kwargs):
        super(BipartitePairData, self).__init__(**kwargs)

        self.inner_edge_index_i, self.inner_edge_index_j = inner_edge_index_i, inner_edge_index_j
        self.x_i, self.x_j = x_j, x_i

        # Allows for passing of non-complete bipartite edges
        if outer_edge_index_i is not None and outer_edge_index_j is not None:
            self.outer_edge_index_i, self.outer_edge_index_j = outer_edge_index_i, outer_edge_index_j
        else:
            self.outer_edge_index_i, self.outer_edge_index_j = self.generate_outer()

    def __inc__(self, key, value):
        if key == 'inner_edge_index_i':
            return self.x_i.size(0)
        if key == 'inner_edge_index_j':
            return self.x_j.size(0)
        if key == 'outer_edge_index_i':
            return torch.tensor([[self.x_i.size(0)], [self.x_j.size(0)]])
        if key == 'outer_edge_index_j':
            return torch.tensor([[self.x_j.size(0)], [self.x_i.size(0)]])
        else:
            return super(BipartitePairData, self).__inc__(key, value)

    def generate_outer(self):
        top = torch.tensor(list(set(range(self.x_i.size(0)))), dtype=torch.long)
        bottom = torch.tensor(list(set(range(self.x_j.size(0)))), dtype=torch.long)
        outer_edge_index_i = torch.cartesian_prod(top, bottom).T
        outer_edge_index_j = torch.cartesian_prod(bottom, top).T
        return outer_edge_index_i, outer_edge_index_j

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x_i is None or self.x_j is None:
            return 0
        if self.x_i.size(1) != self.x_j.size(1):
            raise AttributeError('number of node features must mach for each graph in pair')
        return 1 if self.x_i.dim() == 1 else self.x_j.size(1)
