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
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.data.collate import collate

from GraphCoAttention.data.MultipartiteData import BipartitePairData


class HeteroDrugDrugInteractionData(tg.data.InMemoryDataset, ABC):
    def __init__(self, root):
        self.url = 'http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz'
        self.original_root = root
        super(HeteroDrugDrugInteractionData, self).__init__(root=root)

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
        return 'heterogenous_decagon_ps_ns_V3.pt'

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
        data.edge_attr = torch.from_numpy(graph['edge_feat']).float()
        data.x = torch.from_numpy(graph['node_feat']).float()
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
        raw_tuple = list(records)  # [:1000]

        label_dict = {}
        nondup_tupple = set()
        for cid_i, cid_j, label, name in tqdm(raw_tuple):
            nondup_tupple.add(tuple((cid_i, cid_j)))
            label_dict[tuple((cid_i, cid_j))] = label

        nondup_tupple = list(nondup_tupple)[:1000]

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
            data = tg.data.HeteroData()
            data['x_i'].x = data_i.x.float()
            data['x_j'].x = data_j.x.float()
            data['x_i', 'inner_edge_i', 'x_i'].edge_index = data_i.edge_index.long()
            data['x_i', 'inner_edge_i', 'x_i'].edge_attr = data_i.edge_attr.float()
            data['x_j', 'inner_edge_j', 'x_j'].edge_index = data_j.edge_index.long()
            data['x_j', 'inner_edge_j', 'x_j'].edge_attr = data_j.edge_attr.float()

            data['x_i', 'outer_edge_ij', 'x_j'].edge_index = outer_edge_index_i.long()
            data['x_j', 'outer_edge_ji', 'x_i'].edge_index = outer_edge_index_j.long()
            data['x_i', 'outer_edge_ij', 'x_j'].edge_attr = torch.ones(size=(outer_edge_index_i.max()+1,
                                                                             data_i.edge_attr.size(1)))
            data['x_j', 'inner_edge_j', 'x_j'].edge_attr = torch.ones(size=(outer_edge_index_j.max()+1,
                                                                             data_j.edge_attr.size(1)))

            # data.y = torch.tensor([int(label.strip('C'))], dtype=torch.long)
            data.binary_y = torch.tensor([int(1)], dtype=torch.long)
            data_list.append(data)

        # Negative Sample
        for i in tqdm(range(len(data_tuples))):
            data_i, data_j = random.choice(pyg_molecules), random.choice(pyg_molecules)
            outer_edge_index_i, outer_edge_index_j = self.generate_outer(data_i.x.size(0), data_j.x.size(0))

            data = tg.data.HeteroData()
            data['x_i'].x = data_i.x.float()
            data['x_j'].x = data_j.x.float()
            data['x_i', 'inner_edge_i', 'x_i'].edge_index = data_i.edge_index.long()
            data['x_i', 'inner_edge_i', 'x_i'].edge_attr = data_i.edge_attr.float()
            data['x_j', 'inner_edge_j', 'x_j'].edge_index = data_j.edge_index.long()
            data['x_j', 'inner_edge_j', 'x_j'].edge_attr = data_j.edge_attr.float()

            data['x_i', 'outer_edge_ij', 'x_j'].edge_index = outer_edge_index_i.long()
            data['x_j', 'outer_edge_ji', 'x_i'].edge_index = outer_edge_index_j.long()

            data['x_i', 'outer_edge_ij', 'x_j'].edge_attr = torch.ones(size=(outer_edge_index_i.max()+1,
                                                                             data_i.edge_attr.size(1)))
            data['x_j', 'inner_edge_j', 'x_j'].edge_attr = torch.ones(size=(outer_edge_index_j.max()+1,
                                                                            data_j.edge_attr.size(1)))

            # data.y = torch.tensor([int(label.strip('C'))], dtype=torch.long)
            data.binary_y = torch.tensor([int(0)], dtype=torch.long)
            data_list.append(data)

        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


target = 0


class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = tg.utils.remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class HeteroQM9(tg.data.InMemoryDataset, ABC):
    def __init__(self, root):
        super().__init__(root)
        # self.qm9_dataset_transformed = QM9(root, transform=tg.transforms.Compose([MyTransform(), Complete(),
        #                                          tg.transforms.Distance(norm=False)]))

        self.qm9_dataset = QM9(root=root)

        print(self.mol2pyg('OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O'))

        print(self.qm9_dataset[0])
        print(self.qm9_dataset[0].y)
        print(self.qm9_dataset[0].z)
        print(self.qm9_dataset[0].pos)

        # [print(x.y) for x in self.qm9_dataset]

    @staticmethod
    def mol2pyg(molecule):
        graph = mol.smiles2graph(molecule)
        data = tg.data.Data()
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).float()
        data.x = torch.from_numpy(graph['node_feat']).float()
        return data



if __name__ == '__main__':
    import wandb

    # dataset = HeteroDrugDrugInteractionData(root=os.path.join('GraphCoAttention', 'data'))
    dataset = HeteroQM9(root=os.path.join('GraphCoAttention', 'data'))


    # run = wandb.init(project="flux", entity="syntensor", job_type="dataset-creation")
    # artifact = wandb.Artifact('drug-drug-interaction', type='dataset')
    # artifact.add_reference('s3://syntensor-data/processed')
    # artifact.add_file(os.path.join(os.path.join('GraphCoAttention', 'data', 'processed'),
    #                                'heterogenous_decagon_ps_ns_.pt'))
    # run.log_artifact(artifact)
