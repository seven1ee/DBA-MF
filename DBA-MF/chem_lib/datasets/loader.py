import os
import torch
import json
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    allowable_features = {
        'possible_atomic_num_list' : list(range(1, 119)),
        'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'possible_chirality_list' : [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list' : [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
        ],
        'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'possible_bonds' : [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'possible_bond_dirs' : [ # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
    }
except:
    print('Error rdkit:')
    Chem, AllChem, allowable_features=None,None, None

def mol_to_graph_data_obj_simple(mol,coords):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    if coords is not None:
        pos = torch.tensor(coords, dtype=torch.float)
        # print("coords")
    else:
        print("no coords")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,pos=pos)
    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir,'smiles.csv'),
            header=None).to_numpy()[:,0]



    def get(self, idx):
        data = Data()
        for key in self.data.keys:
        # for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            #smi = self.smiles_list[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        data['smiles']=self.smiles_list[idx]
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)  # ['tox21.json'] #
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def get_non_h_coords(self,mol, mol_h):
        conf = mol_h.GetConformer()
        coords = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            coords.append([pos.x, pos.y, pos.z])
        return torch.tensor(coords, dtype=torch.float)
    def process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels,coords = _load_tox21_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                coord=coords[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol,coord)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels,coords = _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                coord=coords[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol,coord)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels,coords = _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                coord=coords[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol,coord)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, labels, coordinates
    """
    print('input path:'+input_path)
    with open(input_path, 'r') as f:
        data = json.load(f)

    smiles_list = []
    rdkit_mol_objs_list = []
    labels = []
    coords_list = []

    for item in data:
        smi = item["smiles"]
        label = item["label"]
        coords = item["coords"]

        mol = Chem.MolFromSmiles(smi)
        if mol is None or coords is None:
            continue  # 跳过无效分子

        smiles_list.append(smi)
        rdkit_mol_objs_list.append(mol)
        labels.append([label])  # shape: (N, 1)
        coords_list.append(np.array(coords, dtype=np.float32))

    labels = np.array(labels, dtype=int)
    num_neg = np.sum(labels[:, 0] == 0)
    num_pos = np.sum(labels[:, 0] == 1)
    print(f"Loaded {len(smiles_list)} molecules: {num_neg} negative, {num_pos} positive")

    return smiles_list, rdkit_mol_objs_list, labels, coords_list

def _load_muv_dataset(input_path):
    print('input path:'+input_path)

    with open(input_path, 'r') as f:
        data = json.load(f)

    smiles_list = []
    rdkit_mol_objs_list = []
    labels = []
    coords_list = []

    for item in data:
        smi = item["smiles"]
        label = item["label"]
        coords = item["coords"]

        mol = Chem.MolFromSmiles(smi)
        if mol is None or coords is None:
            continue  # 跳过无效分子

        smiles_list.append(smi)
        rdkit_mol_objs_list.append(mol)
        labels.append([label])  # shape: (N, 1)
        coords_list.append(np.array(coords, dtype=np.float32))

    labels = np.array(labels, dtype=int)
    num_neg = np.sum(labels[:, 0] == 0)
    num_pos = np.sum(labels[:, 0] == 1)
    print(f"Loaded {len(smiles_list)} molecules: {num_neg} negative, {num_pos} positive")

    return smiles_list, rdkit_mol_objs_list, labels, coords_list

def _load_sider_dataset(input_path):

    print('input path:'+input_path)

    with open(input_path, 'r') as f:
        data = json.load(f)

    smiles_list = []
    rdkit_mol_objs_list = []
    labels = []
    coords_list = []

    for item in data:
        smi = item["smiles"]
        label = item["label"]
        coords = item["coords"]

        mol = Chem.MolFromSmiles(smi)
        if mol is None or coords is None:
            continue  # 跳过无效分子

        smiles_list.append(smi)
        rdkit_mol_objs_list.append(mol)
        labels.append([label])  # shape: (N, 1)
        coords_list.append(np.array(coords, dtype=np.float32))

    labels = np.array(labels, dtype=int)
    num_neg = np.sum(labels[:, 0] == 0)
    num_pos = np.sum(labels[:, 0] == 1)
    print(f"Loaded {len(smiles_list)} molecules: {num_neg} negative, {num_pos} positive")

    return smiles_list, rdkit_mol_objs_list, labels, coords_list



