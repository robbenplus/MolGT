import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from .data_utils import *
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import pyalgos as algos
from glob import glob
import _pickle as pickle
import torch
from torch.utils.data import Dataset


class Molecular_Feature_Dataset(Dataset):
    def __init__(self, args, mol_list, label_list=[], max_atom=40):
        self.config = args
        self.max_atom = max_atom
        self.mol_list = mol_list
        self.label_list = label_list

    def tokenize_from_smiles(self, smiles):
        if isinstance(smiles, Mol):
            mol = smiles
        else:
            mol = Chem.MolFromSmiles(smiles)

        graph = mol2graph(mol)
        data = preprocess_item(graph)

        attention_mask = attn_mask(data['x'], self.max_atom)

        # pad everything
        data['x'] = pad_x(data['x'], self.max_atom)
        data['in_degree'] = pad_degree(data['in_degree'], self.max_atom, self.config.max_degree)
        data['edge_input'] = pad_edge_input(data['edge_input'], self.max_atom, self.config.multi_hop_max_dist)
        data['spatial_pos'] = pad_spatial_pos(data['spatial_pos'], self.max_atom, self.config.max_spd)
        data['attn_bias'] = pad_attn_bias(data['attn_bias'], self.max_atom)
        data['node_type_edge'] = pair_wise(data['x'], self.max_atom)

        return {'x': data['x'].unsqueeze(0),
                'in_degree': data['in_degree'].unsqueeze(0),
                'edge_input': data['edge_input'].unsqueeze(0),
                'spatial_pos': data['spatial_pos'].unsqueeze(0),
                'attn_bias': data['attn_bias'].unsqueeze(0),
                'attention_mask': attention_mask.unsqueeze(0)}

    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        mol = self.mol_list[index]

        graph = mol2graph(mol)
        data = preprocess_item(graph)

        attention_mask = attn_mask(data['x'], self.max_atom)

        # pad everything
        data['x'] = pad_x(data['x'], self.max_atom)
        data['in_degree'] = pad_degree(data['in_degree'], self.max_atom, self.config.max_degree)
        data['edge_input'] = pad_edge_input(data['edge_input'], self.max_atom, self.config.multi_hop_max_dist)
        data['spatial_pos'] = pad_spatial_pos(data['spatial_pos'], self.max_atom, self.config.max_spd)
        data['attn_bias'] = pad_attn_bias(data['attn_bias'], self.max_atom)

        return {'x': data['x'],
                'in_degree': data['in_degree'],
                'edge_input': data['edge_input'],
                'spatial_pos': data['spatial_pos'],
                'attn_bias': data['attn_bias'],
                'attention_mask': attention_mask}


class Molecular_LMDB_Dataset(Dataset):
    def __init__(self, args, lmdb_env, index_mapping=None, max_atom=40, sample_n=2, mask_ratio=0.15, noise_scale=1.0):
        self.config = args
        self.max_atom = max_atom
        self.sample_n = sample_n
        self.mask_ratio = mask_ratio
        self.noise_scale = noise_scale
        self.txn = lmdb_env['txn']
        self.keys = lmdb_env['keys']

        if index_mapping is None:
            self.index_mapping = torch.arange(len(self.keys))
        else:
            self.index_mapping = index_mapping


    def __len__(self):
        return len(self.index_mapping)


    def __getitem__(self, episodic_index):
        index = self.index_mapping[episodic_index]
        datapoint_pickled = self.txn.get(self.keys[index])
        pkl = pickle.loads(datapoint_pickled)
        
        mol = pkl['mol']
        maccs = pkl['maccs']
        usrcat = pkl['usrcat']
        pos = torch.FloatTensor(mol.GetConformer().GetPositions())

        graph = mol2graph(mol)
        data = preprocess_item(graph)

        # assert(data['x'].shape[0] == pos.shape[0])

        attention_mask = attn_mask(data['x'], self.max_atom)
        maccs = torch.FloatTensor(maccs)
        usrcat = torch.FloatTensor(usrcat)

        # pad everything
        data['x'] = pad_x(data['x'], self.max_atom)
        data['in_degree'] = pad_degree(data['in_degree'], self.max_atom, self.config.max_degree)
        data['edge_input'] = pad_edge_input(data['edge_input'], self.max_atom, self.config.multi_hop_max_dist)
        data['spatial_pos'] = pad_spatial_pos(data['spatial_pos'], self.max_atom, self.config.max_spd)
        data['attn_bias'] = pad_attn_bias(data['attn_bias'], self.max_atom)
        data['node_type_edge'] = pair_wise(data['x'], self.max_atom)
        data['pos'] = pad_3d_pos(pos, self.max_atom)        

        # get atom-motif label
        sample_col_indices = simple_tree_decomp(mol, self.max_atom, self.sample_n)

        # for denoising
        data['pos'], pos_target, pos_mask = generate_noise(data['pos'], data['num_nodes'], 
                                                 self.max_atom, self.mask_ratio, 
                                                 self.noise_scale)

        return {'x': data['x'],
                'in_degree': data['in_degree'],
                'edge_input': data['edge_input'],
                'spatial_pos': data['spatial_pos'],
                'attn_bias': data['attn_bias'],
                'node_type_edge': data['node_type_edge'], 
                'pos': data['pos'],
                'attention_mask': attention_mask,
                'num_atoms': data['num_nodes'],
                'sample_col_indices': sample_col_indices,
                'episodic_index': episodic_index,
                'maccs': maccs,
                'usrcat': usrcat,
                'pos_target': pos_target, 
                'pos_mask': pos_mask}
   


if __name__ == '__main__':
    k = 1
    