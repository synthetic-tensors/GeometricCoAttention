import os
import torch
import numpy as np
from tqdm import tqdm
from abc import ABC, ABCMeta
import torch_geometric as tg
import torch.utils.data.dataset


class DrugDrugInteractionData(tg.data.InMemoryDataset, ABC):
    def __init__(self, root):
        super(DrugDrugInteractionData, self).__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

