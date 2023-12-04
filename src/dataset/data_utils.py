import rdkit.Chem as Chem
import torch
import random
import numpy as np
import pandas as pd
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import pyalgos as algos
import os
from glob import glob
import _pickle as pickle
import torch
from torch.utils.data import Dataset, DataLoader
from .atom_features import atom_to_feature_vector, bond_to_feature_vector

import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def mol2graph(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_attr'] = edge_attr
    graph['x'] = x
    graph['num_nodes'] = len(x)

    return graph 


def preprocess_item(item):
        edge_attr, edge_index, x = item['edge_attr'], item['edge_index'], item['x']
        N = x.shape[0]

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.shape[-1]], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = torch.LongTensor(edge_attr)

        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

        # combine
        item['x'] = torch.LongTensor(x)
        # item['adj'] = adj
        item['attn_bias'] = attn_bias
        # item['attn_edge_type'] = attn_edge_type
        item['spatial_pos'] = spatial_pos
        item['in_degree'] = adj.long().sum(dim=1).view(-1)
        # item['out_degree'] = adj.long().sum(dim=0).view(-1)
        item['edge_input'] = torch.from_numpy(edge_input).long()
        item['num_nodes'] = torch.LongTensor([item['num_nodes']])

        return item


def pad_x(x, max_atom):
    x = x + 1
    x = x[:max_atom, :]
    padding_num = max_atom - x.shape[0]
    paddings = torch.zeros(padding_num, x.shape[1], dtype=torch.long)
    new_x = torch.cat([x, paddings], dim=0)

    return new_x


def pad_degree(degree, max_atom, max_degree):
    degree = degree[:max_atom] + 1  # 0 for padding, some mols hava atoms with 0 degree 
    degree = torch.clamp(degree, 0, max_degree - 1)
    padding_num = max_atom - degree.shape[0]
    paddings = torch.zeros(padding_num, dtype=torch.long)
    new_degree = torch.cat([degree, paddings])

    return new_degree


def pad_edge_input(edge_input, max_atom, multi_hop_max_dist):
    # edge_input: [n_node, n_node, max_dist, edge]
    edge_input = edge_input + 1  # 0 for padding token
    edge_input = edge_input[:max_atom, :max_atom, :multi_hop_max_dist, :]

    atom_padding_num = max_atom - edge_input.shape[0]

    # pad dim 0
    paddings = torch.zeros(atom_padding_num, edge_input.shape[1], edge_input.shape[2], edge_input.shape[3], dtype=torch.long)
    edge_input = torch.cat([edge_input, paddings], dim=0)

    # pad dim 1
    paddings = torch.zeros(edge_input.shape[0], atom_padding_num, edge_input.shape[2], edge_input.shape[3], dtype=torch.long)
    edge_input = torch.cat([edge_input, paddings], dim=1)

    # pad dim 2
    dist_padding_num = multi_hop_max_dist - edge_input.shape[2]
    # if dist_padding_num < 0:
    #     print(edge_input.shape)
    paddings = torch.zeros(edge_input.shape[0], edge_input.shape[1], dist_padding_num, edge_input.shape[3], dtype=torch.long)
    edge_input = torch.cat([edge_input, paddings], dim=2)

    return edge_input


def pad_spatial_pos(spatial_pos, max_atom, max_spd):
    # spatial_pos: [n_node, n_node]
    spatial_pos = spatial_pos[:max_atom, :max_atom] + 1  # padding is 0

    spatial_pos = torch.clamp(spatial_pos, 0, max_spd - 1)

    padding_num = max_atom - spatial_pos.shape[0]

    # pad dim 0
    paddings = torch.zeros(padding_num, spatial_pos.shape[0], dtype=torch.long)
    spatial_pos = torch.cat([spatial_pos, paddings], dim=0)

    # pad dim 1
    paddings = torch.zeros(spatial_pos.shape[0], padding_num, dtype=torch.long)
    spatial_pos = torch.cat([spatial_pos, paddings], dim=1)

    return spatial_pos


def pad_attn_bias(attn_bias, max_atom):
    # attn_bias: [n_node+1, n_node+1]
    attn_bias = attn_bias[:max_atom + 1, :max_atom + 1]

    padding_num = max_atom - attn_bias.shape[0] + 1  # +1 for graph_token

    # pad dim 0
    paddings = torch.zeros(padding_num, attn_bias.shape[0], dtype=torch.long)
    attn_bias = torch.cat([attn_bias, paddings], dim=0)

    # pad dim 1
    paddings = torch.zeros(attn_bias.shape[0], padding_num, dtype=torch.long)
    attn_bias = torch.cat([attn_bias, paddings], dim=1)

    return attn_bias


def pair_wise(x, max_atom):
    # x: [max_atom, num_feature]
    # [max_atom]
    node_atom_type = x[:, 0]  
    # [max_atom+1]
    node_atom_type = torch.cat([torch.LongTensor([149]), node_atom_type])  
    node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, max_atom+1).unsqueeze(-1)
    node_atom_j = node_atom_type.unsqueeze(0).repeat(max_atom+1, 1).unsqueeze(-1)
    node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)

    return node_atom_edge


def pad_3d_pos(pos, max_atom):
    if pos == None:
        return None
    # pos: [max_atom, 3]
    pos = pos[:max_atom, :]
    global_pos = pos.mean(dim=0)

    # [max_atom+1, 3]
    pos = torch.cat([global_pos.unsqueeze(0), pos], dim=0)

    padding_num = max_atom + 1 - pos.shape[0]
    paddings = torch.zeros(padding_num, 3, dtype=torch.float32)

    pos = torch.cat([pos, paddings], dim=0)

    return pos


def attn_mask(x, max_atom):
    attention_mask = torch.zeros(max_atom + 1, dtype=torch.long)  # +1 for graph token
    attention_mask[:x.shape[0] + 1] = 1

    return attention_mask


def simple_tree_decomp(mol, max_atoms, sample_n):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return torch.zeros(max_atoms, sample_n, dtype=torch.long)

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)
    
    # each atom's neibors: [atoms, neis]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(n_atoms):
        nei = []
        for c in cliques:
            if i in c:
                nei.extend(c)
        nei = set(nei)
        if len(nei) == 0:
            nei = [random.randint(0, n_atoms-1)]
            nei_list[i] = nei
        else:
            nei.discard(i)
            nei_list[i] = list(nei)

    sample_col_indices = []
    for nei in nei_list:    
        res = random.choices(nei, k=sample_n)
        sample_col_indices.append(res)

    sample_col_indices = sample_col_indices[:max_atoms]
    paddings = torch.zeros(max_atoms - len(sample_col_indices), sample_n, dtype=torch.long)
    sample_col_indices = torch.LongTensor(sample_col_indices)
    sample_col_indices = torch.cat([sample_col_indices, paddings], dim=0)
    sample_col_indices.clamp_(0, max_atoms - 1)
            
    return sample_col_indices


def postive_random_sampler(batch_size, max_atoms, nei_list, sample_n=1):
    sample_shape_perrow = (1, max_atoms, sample_n)
    sample_row_indices = [torch.full(sample_shape_perrow, row_id, dtype=torch.long) \
                            for row_id in range(batch_size)]
    sample_row_indices = torch.cat(sample_row_indices, 0)

    sample_col_indices = []

    for nei in nei_list:    
        res = random.choices(nei, k=sample_n)
        sample_col_indices.append(res)

    sample_col_indices = sample_col_indices[:max_atoms]
    paddings = torch.zeros(max_atoms - len(sample_col_indices), sample_n, dtype=torch.long)
    sample_col_indices = torch.LongTensor(sample_col_indices)


def generate_noise(pos, num_atoms, max_atom=40, mask_ratio=0.15, noise_scale=1.0):
    # choose where to mask
    # pos: [nodes+1, 3]
    num_atoms = num_atoms if num_atoms <= max_atom else max_atom
    num_mask = int(mask_ratio*num_atoms) if int(mask_ratio*num_atoms) else 1
    mask_idx = np.random.choice(np.arange(1, num_atoms+1), num_mask, replace=False).tolist()  # index 0 is graph token

    # add noise
    noise = np.random.uniform(-noise_scale, noise_scale, (len(mask_idx), 3))
    noise = torch.FloatTensor(noise)
    pos[mask_idx] += noise

    # get target
    pos_target = torch.zeros((max_atom+1, 3), dtype=torch.float32)
    pos_target[mask_idx] = noise

    # get mask
    pos_mask = torch.zeros_like(pos_target).bool()
    pos_mask[mask_idx] = 1

    return pos, pos_target, pos_mask

def create_dataloader(dataset, batch_size=32, workers=0, pin_memory=True, drop_last=False, shuffle=False, sampler=None):
    if sampler is not None:
        shuffle = False

    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=workers, pin_memory=pin_memory,
                      drop_last=drop_last, shuffle=shuffle, sampler=sampler)