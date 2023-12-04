from .data_utils import *
from deepchem.molnet import *
import os
import torch
from torch.utils.data import Dataset
import deepchem as dc


class Molnet_Classification_Dataset(Dataset):
    def __init__(self, args, name='bbbp', mode='train'):
        self.config = args
        self.mode = mode
        self.data_dir = os.path.join(args.molnet_path, name)
        if name == 'bbbp':
            _, splits, _ = load_bbbp(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir)
        elif name == 'bace':
            _, splits, _ = load_bace_classification(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir)
        elif name == 'clintox':
            _, splits, _ = load_clintox(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir)
        elif name == 'tox21':
            _, splits, _ = load_tox21(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir)
        elif name == 'toxcast':
            _, splits, _ = load_toxcast(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir)
        elif name == 'sider':
            _, splits, _ = load_sider(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir)
        else:
            raise ValueError

        train, valid, test = splits
        self.mol_list = []
        self.w_list = []
        self.label_list = []
        self.max_atom = 0

        if mode == 'train':
            num_list = []
            for mol, label, w, smiles in train.itersamples():
                self.mol_list.append(mol)
                num_list.append(mol.GetNumAtoms())
                self.label_list.append(label)
                self.w_list.append(w)
            self.max_atom = int(max(num_list))

        elif mode == 'valid':
            num_list = []
            for mol, label, w, smiles in valid.itersamples():
                self.mol_list.append(mol)
                num_list.append(mol.GetNumAtoms())
                self.label_list.append(label)
                self.w_list.append(w)
            self.max_atom = int(max(num_list))

        elif mode == 'test':
            num_list = []
            for mol, label, w, smiles in test.itersamples():
                self.mol_list.append(mol)
                num_list.append(mol.GetNumAtoms())
                self.label_list.append(label)
                self.w_list.append(w)
            self.max_atom = int(max(num_list))

        else:
            raise ValueError

        print(f'mode {self.mode} , max atoms {self.max_atom}, mols {len(self.mol_list)}')
        self.max_atom = self.max_atom if self.max_atom <= 135 else 135  # 135 for others, 264 for sider
        # print(sorted(num_list, reverse=True)[:20])
        # print(self.label_list)

    def __len__(self):
        return len(self.mol_list)

    @property
    def total_samples(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        mol = self.mol_list[index]
        graph = mol2graph(mol)
        data = preprocess_item(graph)

        attention_mask = attn_mask(data['x'], self.max_atom)
        label = torch.FloatTensor(self.label_list[index])
        w = torch.FloatTensor(self.w_list[index])

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
                'attention_mask': attention_mask,
                'w': w,
                'label': label}


class Molnet_Regression_Dataset(Dataset):
    def __init__(self, args, name='esol', mode='train'):
        self.config = args
        self.mode = mode
        self.data_dir = os.path.join(args.molnet_path, name)
        if name == 'esol':
            _, splits, _ = load_delaney(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir, transformers=[])
        elif name == 'freesolv':
            _, splits, _ = load_freesolv(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir, transformers=[])
        elif name == 'lipo':
            _, splits, _ = load_lipo(featurizer="Raw", splitter='scaffold', data_dir=self.data_dir, save_dir=self.data_dir, transformers=[])
        else:
            raise ValueError

        train, valid, test = splits
        self.mol_list = []
        self.w_list = []
        self.label_list = []
        self.max_atom = 0

        if mode == 'train':
            num_list = []
            for mol, label, w, smiles in train.itersamples():
                self.mol_list.append(mol)
                num_list.append(mol.GetNumAtoms())
                self.label_list.append(label)
                self.w_list.append(w)
            self.max_atom = int(max(num_list))

        elif mode == 'valid':
            num_list = []
            for mol, label, w, smiles in valid.itersamples():
                self.mol_list.append(mol)
                num_list.append(mol.GetNumAtoms())
                self.label_list.append(label)
                self.w_list.append(w)
            self.max_atom = int(max(num_list))

        elif mode == 'test':
            num_list = []
            for mol, label, w, smiles in test.itersamples():
                self.mol_list.append(mol)
                num_list.append(mol.GetNumAtoms())
                self.label_list.append(label)
                self.w_list.append(w)
            self.max_atom = int(max(num_list))

        else:
            raise ValueError

        print(f'mode {self.mode} , max atoms {self.max_atom}, mols {len(self.mol_list)}')
        # self.max_atom = self.max_atom if self.max_atom <= 128 else 128

    def __len__(self):
        return len(self.mol_list)

    @property
    def total_samples(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        mol = self.mol_list[index]
        graph = mol2graph(mol)
        data = preprocess_item(graph)

        attention_mask = attn_mask(data['x'], self.max_atom)
        label = torch.FloatTensor(self.label_list[index])

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
                'attention_mask': attention_mask,
                'label': label}


class Regression_Dataset(Dataset):
    def __init__(self, args, df, name='malaria', mode='train'):
        self.config = args
        self.mode = mode
        self.df = df
        self.name = name
        if name == 'malaria':
            self.label_list = df['activity'].values
        elif name == 'cep':
            self.label_list = df['PCE'].values

        self.smi_list = df['smiles'].values
        self.mol_list = []
        self.num_list = []

        for smi in self.smi_list:
            mol = Chem.MolFromSmiles(smi)
            self.mol_list.append(mol)
            self.num_list.append(mol.GetNumAtoms())

        self.mol_list = np.array(self.mol_list)
        self.num_list = np.array(self.num_list)

        dataset = dc.data.DiskDataset.from_numpy(X=self.label_list, y=self.label_list, ids=self.smi_list)
        scaffold_splitter = dc.splits.ScaffoldSplitter()
        train_idx, valid_idx, test_idx = scaffold_splitter.split(dataset)

        if mode == 'train':
            self.mol_list = self.mol_list[train_idx]
            self.label_list = self.label_list[train_idx]
            self.max_atom = int(max(self.num_list[train_idx]))

        elif mode == 'valid':
            self.mol_list = self.mol_list[valid_idx]
            self.label_list = self.label_list[valid_idx]
            self.max_atom = int(max(self.num_list[valid_idx]))

        elif mode == 'test':
            self.mol_list = self.mol_list[test_idx]
            self.label_list = self.label_list[test_idx]
            self.max_atom = int(max(self.num_list[test_idx]))

        else:
            raise ValueError

        print(f'mode {self.mode} , max atoms {self.max_atom}, mols {len(self.mol_list)}')
        # self.max_atom = self.max_atom if self.max_atom <= 128 else 128

    def __len__(self):
        return len(self.mol_list)

    @property
    def total_samples(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        mol = self.mol_list[index]
        graph = mol2graph(mol)
        data = preprocess_item(graph)

        attention_mask = attn_mask(data['x'], self.max_atom)
        label = torch.FloatTensor([self.label_list[index]])

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
                'attention_mask': attention_mask,
                'label': label}
    



if __name__ == '__main__':
    x = 1
    
    