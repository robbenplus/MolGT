import os
from glob import glob
import torch
from torch import distributed as dist
import torch.distributed.nn


def row_sample(batch_size, max_atoms, sample_n=1):
    sample_shape_perrow = (1, max_atoms, sample_n)

    sample_row_indices = [torch.full(sample_shape_perrow, row_id, dtype=torch.long) \
                            for row_id in range(batch_size)]
    
    sample_row_indices = torch.cat(sample_row_indices, 0)
    
    return sample_row_indices


def fast_negative_sample(batch_size, max_atoms, sample_n=50):
    sample_shape_perrow = (1, max_atoms, sample_n)
    sample_row_indices = [(torch.randint(batch_size, sample_shape_perrow, dtype=torch.long)+row_id+1)%batch_size \
                            for row_id in range(batch_size)]
    sample_row_indices = torch.cat(sample_row_indices) 
    
    sample_shape_allcol = (batch_size, max_atoms, sample_n)
    sample_col_indices = torch.randint(max_atoms, sample_shape_allcol, dtype=torch.long)

    return sample_row_indices, sample_col_indices


def negative_sample(batch_size, max_atoms, num_atoms, sample_n=50):
    sample_shape_perrow = (1, max_atoms, sample_n)
    sample_row_indices = [(torch.randint(batch_size-1, sample_shape_perrow, dtype=torch.long)+row_id+1)%batch_size \
                            for row_id in range(batch_size)]
    sample_row_indices = torch.cat(sample_row_indices) 
    
    col_list = []
    for idx in sample_row_indices.reshape(-1):
        atoms = num_atoms[idx]
        col = torch.randint(int(atoms), [1])
        col_list.append(col)

    sample_col_indices = torch.stack(col_list).reshape(batch_size, max_atoms, sample_n)

    return sample_row_indices, sample_col_indices

