This is a Pytorch implementation of the `Pre-training Graph Transformer for Molecular Representation`paper:



## Environment

Create a new environment with all required packages using `environment.yml` (this can take a while). While in the project directory run:

```shell
conda env create -f environment.yml
```

Activate the environment

```shell
conda activate 3DGraphormer
```

## Data

We use LMDB to store data. For pre-training, we provide an example LMDB file in /data/example_data.lmdb. Each value contains three items: rd_mol, maccs and usrcat fingerprint.

You can use the following code snippets to read from the LMDB file:

```python
lmdb_path = '.../example_data.lmdb'
    
env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=512,
    )
txn = env.begin()
keys = list(txn.cursor().iternext(values=False))

datapoint_pickled = txn.get(keys[0])
pkl_data = pickle.loads(datapoint_pickled)

print(pkl_data['mol'])
print(pkl_data['maccs'])
print(pkl_data['usrcat'])
```


## Pre-trained Model

We provide the MolGT pre-trained model at `src/ckpt/pytorch_model.bin`.



## Pre-train a model

```shell
CUDA_VISIBLE_DEVICES="0,1" python -u -m torch.distributed.launch --nproc_per_node 2 episode_PT.py
```



## Fine-tune a model

Our results in the paper can be reproduced using seed 1, 2, 3 with scaffold splitting. (RTX3090 + CUDA11.3 + PyTorch1.11.0)

Fine-tuning for downstream tasks might lead to results slightly different from those reported in the paper, due to differences in CUDA and GPU versions (We observed in RTX2080Ti + CUDA10.2).

For Classification:

```shell
python finetune_classification.py --seed=1 --classification_dataset=bbbp
```

For Regression

```shell
python finetune_regression.py --seed=1 --regression_dataset=esol
```

## Molecular Feature Extraction
We can use the following code to obtain node and graph representations by MolGT on customized data: 

```python
import torch
from dataset import Molecular_Feature_Dataset
from models import GraphormerModel, GraphormerConfig
from config import args

# prepare data
dataset = Molecular_Feature_Dataset(args, [])
data = dataset.tokenize_from_smiles('CCC')

# load pre-trained model
gconf = GraphormerConfig()
model = GraphormerModel(gconf)
model.load_state_dict(torch.load('./ckpt/pytorch_model.bin'), strict=False)

# get node and graph representations
node, graph = model(data, mode='2d')
print(node.shape)  # [1, max_atom, hidden]
print(graph.shape) # [1, hidden]
```