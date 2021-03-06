import os
import torch
import wget
import wandb
import itertools
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC, ABCMeta
from retrying import retry

from ogb.utils.url import decide_download, download_url, extract_zip
import ogb.utils.mol as mol
import torch_geometric as tg
from torch_geometric.data.collate import collate

from GraphCoAttention.data.MultipartiteData import BipartitePairData


class DrugDrugInteractionData(tg.data.InMemoryDataset, ABC):
    def __init__(self, root):
        self.url = 'http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz'
        self.original_root = root
        super(DrugDrugInteractionData, self).__init__(root=root)

        run = wandb.init(entity="syntensor", project="flux")
        ordinal_artifact = run.use_artifact('syntensor/flux/drug-drug-interaction:latest', type='dataset')
        # ordinal_path = ordinal_artifact.get_path(self.processed_paths[0])
        # ordinal_path.download(root=self.processed_dir)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'bio-decagon-combo.tar.gz'

    @property
    def processed_file_names(self):
        return 'decagon_ps_ns_full.pt'

    def download(self):
        if decide_download(self.url):
            wget.download(self.url, self.raw_paths[0])
        else:
            print('Stop download.')
            exit(-1)

    @staticmethod
    def mol2pyg(molecule):
        graph = mol.smiles2graph(molecule)
        data = tg.data.Data()
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        return data

    @staticmethod
    def generate_outer(x_i_size, x_j_size):
        top = torch.tensor(list(set(range(x_i_size))), dtype=torch.long)
        bottom = torch.tensor(list(set(range(x_j_size))), dtype=torch.long)
        outer_edge_index_i = torch.cartesian_prod(top, bottom).T
        outer_edge_index_j = torch.cartesian_prod(bottom, top).T
        return outer_edge_index_i, outer_edge_index_j

    @retry(wait_random_min=3000, wait_random_max=4000)
    def pubchem_cid_smiles(self, cid_i, cid_j):
        import pubchempy as pcp
        mol_i = pcp.Compound.from_cid(int(cid_i.strip('CID'))).isomeric_smiles
        mol_j = pcp.Compound.from_cid(int(cid_j.strip('CID'))).isomeric_smiles
        return mol_i, mol_j

    def process(self):

        df = pd.read_csv(self.raw_paths[0], compression='gzip', header=0, encoding="ISO-8859-1", error_bad_lines=False)

        print(df)

        records = df.to_records(index=False)
        raw_tuple = list(records)  # [:20]

        label_dict = {}
        nondup_tupple = set()
        for cid_i, cid_j, label, name in tqdm(raw_tuple):
            nondup_tupple.add(tuple((cid_i, cid_j)))
            label_dict[tuple((cid_i, cid_j))] = label

        nondup_tupple = list(nondup_tupple)

        data_list = []
        data_tuples = []
        pos = {}
        data_label_dict = {}
        # Generate PyG representations of molecules
        for cid_i, cid_j in tqdm(nondup_tupple):
            mol_i, mol_j = self.pubchem_cid_smiles(cid_i, cid_j)
            data_i, data_j = self.mol2pyg(mol_i), self.mol2pyg(mol_j)
            data_tuples.append((data_i, data_j))
            data_label_dict[tuple((data_i, data_j))] = label_dict[tuple((cid_i, cid_j))]
            pos[data_i] = data_j
            pos[data_j] = data_i

        pyg_molecules = list(itertools.chain.from_iterable(data_tuples))
        torch.save(pyg_molecules, os.path.join(self.processed_dir, 'pyg_molecules.pt'))

        # Positive Sample
        for data_i, data_j in tqdm(data_tuples):
            label = data_label_dict[tuple((data_i, data_j))]
            outer_edge_index_i, outer_edge_index_j = self.generate_outer(data_i.x.size(0), data_j.x.size(0))

            data = BipartitePairData(x_i=data_i.x, x_j=data_j.x,
                                     inner_edge_index_i=data_i.edge_index,
                                     inner_edge_index_j=data_j.edge_index,
                                     outer_edge_index_i=outer_edge_index_i,
                                     outer_edge_index_j=outer_edge_index_j,
                                     y=torch.tensor([int(label.strip('C'))], dtype=torch.long),
                                     binary_y=torch.tensor([int(1)], dtype=torch.long))

            data_list.append(data)

        # Negative Sample
        for i in tqdm(range(len(data_tuples))):
            mol_i, mol_j = random.choice(pyg_molecules), random.choice(pyg_molecules)
            outer_edge_index_i, outer_edge_index_j = self.generate_outer(mol_i.x.size(0), mol_j.x.size(0))

            data = BipartitePairData(x_i=mol_i.x, x_j=mol_j.x,
                                     inner_edge_index_i=mol_i.edge_index,
                                     inner_edge_index_j=mol_j.edge_index,
                                     outer_edge_index_i=outer_edge_index_i,
                                     outer_edge_index_j=outer_edge_index_j,
                                     y=torch.tensor([int(0)], dtype=torch.long),
                                     binary_y=torch.tensor([int(0)], dtype=torch.long))

            data_list.append(data)

        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def collate(data_list):
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
            follow_batch=['x_i', 'x_j']
        )
        return data, slices


if __name__ == '__main__':
    import wandb
    dataset = DrugDrugInteractionData(root=os.path.join('GraphCoAttention', 'data'))
    run = wandb.init(project="flux", entity="syntensor", job_type="dataset-creation")
    artifact = wandb.Artifact('drug-drug-interaction', type='dataset')
    artifact.add_reference('s3://syntensor-data/processed')
    artifact.add_file(os.path.join(os.path.join('GraphCoAttention', 'data', 'processed'), 'decagon_ps_ns_full.pt'))
    run.log_artifact(artifact)
    