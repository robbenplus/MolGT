import torch
import warnings
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import math
from torch.cuda import amp
from distributed import *
import torch.distributed as dist
import gc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_acc(y_true, y_score):
    acc = y_true.eq(y_score.argmax(-1)).sum() / y_true.shape[0]

    return acc

def feature_extraction(model, dataloader, epoch, clustering, args, mode='fp'):
    # mode: fp or projection
    model.eval()
    if isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    for _, batched_data in enumerate(dataloader):
        # batched_data = dict()
        for key in batched_data.keys():
            batched_data[key] = batched_data[key].cuda()

        with torch.no_grad(), amp.autocast():
            graph_2d, graph_3d = model(batched_data, feature=True)
        
        index = batched_data['episodic_index']
        if args.distributed:
            index = concat_all_gather(index, gather_with_grad=False)
            graph_2d = concat_all_gather(graph_2d, gather_with_grad=False)
            graph_3d = concat_all_gather(graph_3d, gather_with_grad=False)
            #dist.barrier() # 1

        if mode == 'fp':
            maccs, usrcat = batched_data['maccs'], batched_data['usrcat'] 
            if args.distributed:
                maccs = concat_all_gather(maccs, gather_with_grad=False)
                usrcat = concat_all_gather(usrcat, gather_with_grad=False)
            if is_master(args):
                clustering.load_batch_fp(index, maccs, usrcat, graph_2d, graph_3d)
                # clustering.load_batch(index, graph_2d, graph_3d)
                #dist.barrier() # 2
        elif mode == 'projection':
            if is_master(args):
                maccs = 1
                usrcat = 1
                clustering.load_batch(index, graph_2d, graph_3d)
                #dist.barrier() # 3
