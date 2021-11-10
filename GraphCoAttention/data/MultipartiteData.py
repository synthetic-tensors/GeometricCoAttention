from typing import (Optional, Dict, Any)
import torch_geometric as tg
import torch

from torch_geometric.typing import OptTensor
from torch_geometric.data.storage import (BaseStorage, NodeStorage,
                                          EdgeStorage, GlobalStorage)


class BipartitePairData(tg.data.Data):
    def __init__(self,
                 inner_edge_index_i:  OptTensor = None, x_i:  OptTensor = None, outer_edge_index_i:  OptTensor = None,
                 inner_edge_index_j:  OptTensor = None, x_j:  OptTensor = None, outer_edge_index_j:  OptTensor = None,
                 **kwargs):
        super(BipartitePairData, self).__init__(**kwargs)

        self.__dict__['_store'] = GlobalStorage(_parent=self)

        self.x_i = x_i
        self.x_j = x_j

        # self.num_nodes = int(x_i.size(0) + x_j.size(0))

        self.inner_edge_index_i = inner_edge_index_i
        self.inner_edge_index_j = inner_edge_index_j

        self.outer_edge_index_i = outer_edge_index_i
        self.outer_edge_index_j = outer_edge_index_j

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key: str, value: Any, *args, **kwargs):
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

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs):
        if key == 'x_i' or key == 'x_j':
            return 0
        elif 'index' in key:
            return 1
        else:
            return 0

    @staticmethod
    def generate_outer(x_i_size, x_j_size):
        top = torch.tensor(list(set(range(x_i_size))), dtype=torch.long)
        bottom = torch.tensor(list(set(range(x_j_size))), dtype=torch.long)
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

    @property
    def num_nodes(self):
        r"""Returns the number of nodes in both graphs."""
        return int(self.x_i.size(0) + self.x_j.size(0))

    # @property
    # def x_i(self):
    #     return self['x_i'] if 'x_i' in self._store else None
    #
    # @property
    # def x_j(self):
    #     return self['x_i'] if 'x_i' in self._store else None
    #
    # @property
    # def inner_edge_index_i(self):
    #     return self['inner_edge_index_i'] if 'inner_edge_index_i' in self._store else None
    #
    # @property
    # def inner_edge_index_j(self):
    #     return self['inner_edge_index_j'] if 'inner_edge_index_j' in self._store else None
    #
    # @property
    # def outer_edge_index_i(self):
    #     return self['outer_edge_index_i'] if 'outer_edge_index_i' in self._store else None
    #
    # @property
    # def outer_edge_index_j(self):
    #     return self['outer_edge_index_j'] if 'outer_edge_index_j' in self._store else None
    #
    # @property
    # def y(self):
    #     return self['y'] if 'y' in self._store else None
