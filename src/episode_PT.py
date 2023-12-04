from config import args
from queue import Queue
from dataset import *
from distributed import *
from clustering import Clustering
from utils import epoch_time, seed_everything, feature_extraction
import numpy as np
import os
import copy
import time
from tqdm import tqdm
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda import amp
from transformers import get_cosine_schedule_with_warmup
from models import GraphormerConfig, GraphormerModelForPretraining
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import lmdb


scaler = amp.GradScaler()
ckpts = Queue(maxsize = args.max_ckpts)


def train_epoch(model, optimizer, train_dataloader, epoch):
    train_dataloader.sampler.set_epoch(epoch)
    start_time = time.time()
    model.train()

    total_loss = 0
    total_loss_CL = 0
    total_loss_infomotif_2d = 0
    total_loss_denoise = 0
    total_loss_proto2d_1 = 0
    total_loss_proto2d_2 = 0
    total_loss_proto2d_3 = 0
    total_loss_proto3d_1 = 0
    total_loss_proto3d_2 = 0
    total_loss_proto3d_3 = 0

    CL_acc_accum = 0
    im2d_acc_accum = 0
    proto2d_acc1_accum = 0
    proto2d_acc2_accum = 0
    proto2d_acc3_accum = 0
    proto3d_acc1_accum = 0
    proto3d_acc2_accum = 0
    proto3d_acc3_accum = 0

    for idx, batch in enumerate(train_dataloader):
        batched_data = dict()
        for key in batch.keys():
            try:
                batched_data[key] = batch[key].cuda()
            except:
                pass

        if args.use_amp and epoch < args.stop_amp_epoch:
            with amp.autocast():
                loss_CL, loss_infomotif_2d, loss_denoise, loss_proto2d_1, loss_proto2d_2, loss_proto2d_3,\
                loss_proto3d_1, loss_proto3d_2, loss_proto3d_3,\
                CL_acc, im2d_acc, proto2d_acc_1, proto2d_acc_2, proto2d_acc_3,\
                proto3d_acc_1, proto3d_acc_2, proto3d_acc_3 = model(batched_data, feature=False, clustering=clustering)
                
                loss = loss_CL + loss_infomotif_2d + loss_denoise + loss_proto2d_1 + loss_proto2d_2 + loss_proto2d_3 + loss_proto3d_1 + loss_proto3d_2 + loss_proto3d_3
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            break

        total_loss += loss.item()  
        total_loss_CL += loss_CL.item() 
        total_loss_infomotif_2d += loss_infomotif_2d.item() 
        total_loss_denoise += loss_denoise.item()
        total_loss_proto2d_1 += loss_proto2d_1.item()
        total_loss_proto2d_2 += loss_proto2d_2.item()
        total_loss_proto2d_3 += loss_proto2d_3.item()
        total_loss_proto3d_1 += loss_proto3d_1.item()
        total_loss_proto3d_2 += loss_proto3d_2.item()
        total_loss_proto3d_3 += loss_proto3d_3.item()

        CL_acc_accum += CL_acc
        im2d_acc_accum += im2d_acc
        proto2d_acc1_accum += proto2d_acc_1
        proto2d_acc2_accum += proto2d_acc_2
        proto2d_acc3_accum += proto2d_acc_3
        proto3d_acc1_accum += proto3d_acc_1
        proto3d_acc2_accum += proto3d_acc_2
        proto3d_acc3_accum += proto3d_acc_3

        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

        scheduler.step()
        model.zero_grad()

        if is_master(args):
            if (idx + 1) % args.log_step == 0 or (idx + 1) == len(train_dataloader):
                print(f'Step {idx + 1}/{len(train_dataloader)} | '
                    f'Total loss: {total_loss / (idx + 1):.1f} | '
                    f'CL loss: {total_loss_CL / (idx + 1):.2f} | '
                    f'im loss: {total_loss_infomotif_2d / (idx + 1):.2f} | '
                    f'denoise loss: {total_loss_denoise / (idx + 1):.5f} | '
                    f'proto2d loss1: {total_loss_proto2d_1 / (idx + 1):.4f} | '
                    f'proto2d loss2: {total_loss_proto2d_2 / (idx + 1):.4f} | '
                    f'proto2d loss3: {total_loss_proto2d_3 / (idx + 1):.4f} | '
                    f'proto3d loss1: {total_loss_proto3d_1 / (idx + 1):.4f} | '
                    f'proto3d loss2: {total_loss_proto3d_2 / (idx + 1):.4f} | '
                    f'proto3d loss3: {total_loss_proto3d_3 / (idx + 1):.4f} | '
                    f'proto2d acc1: {(proto2d_acc1_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'proto2d acc2: {(proto2d_acc2_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'proto2d acc3: {(proto2d_acc3_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'proto3d acc1: {(proto3d_acc1_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'proto3d acc2: {(proto3d_acc2_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'proto3d acc3: {(proto3d_acc3_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'CL acc: {(CL_acc_accum / (idx + 1)) * 100:.3f}% | '
                    f'im acc: {(im2d_acc_accum / (idx + 1)) * 100:.3f}% | '
                    f'Time: {((time.time() - start_time) / 60.0):.1f}min | '
                    f'lr: {cur_lr:g}')

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if is_master(args):
        proto2d_acc = (proto2d_acc1_accum + proto2d_acc2_accum + proto2d_acc3_accum) / 3.0 / len(train_dataloader) * 100
        proto3d_acc = (proto3d_acc1_accum + proto3d_acc2_accum + proto3d_acc3_accum) / 3.0 / len(train_dataloader) * 100
        print(' ')
        print(f'Epoch {epoch + 1} time: {epoch_mins}m {epoch_secs}s train loss: {total_loss / len(train_dataloader):.4f}')
        print(f'CL acc: {CL_acc_accum / len(train_dataloader):.6f}  im acc: {im2d_acc_accum / len(train_dataloader):.6f}  ' 
              f'proto2d acc: {proto2d_acc:.6f}%  proto3d acc: {proto3d_acc:.6f}%')
        print(' ')

    return total_loss / len(train_dataloader)


def train(model, optimizer, dataloader):  

    best_loss = 99999
    for epoch in range(args.train_epoch):

        all_index_mapping[:] = torch.from_numpy(np.random.choice(args.dataset_size, args.episode_size, replace=False))

        clustering.reset()

        if epoch < args.fp_epoch or (epoch + 1) % 10 == 0:
            start_time = time.time()

            feature_extraction(model, dataloader, epoch, clustering, args, mode='fp')
            if is_master(args):
                print('generate_labels_from_fp...')
                clustering.generate_labels_from_fp()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'time: {epoch_mins}:{epoch_secs}')

        else: 
            start_time = time.time()

            feature_extraction(model, dataloader, epoch, clustering, args, mode='projection')
            if is_master(args):
                print('generate_labels...')
                clustering.generate_labels()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'time: {epoch_mins}:{epoch_secs}')

        dist.barrier()
        clustering.sync_prototypes(args)

        assert (clustering.mol_2d_labels_1 == -1).sum().item() == 0, 'error'
        assert (clustering.mol_2d_labels_2 == -1).sum().item() == 0, 'error'
        assert (clustering.mol_2d_labels_3 == -1).sum().item() == 0, 'error'
        assert (clustering.mol_3d_labels_1 == -1).sum().item() == 0, 'error'
        assert (clustering.mol_3d_labels_2 == -1).sum().item() == 0, 'error'
        assert (clustering.mol_3d_labels_3 == -1).sum().item() == 0, 'error'
        assert (torch.isnan(clustering.mol_2d_centroids_1)).sum().item() == 0, 'error'
        assert (torch.isnan(clustering.mol_2d_centroids_2)).sum().item() == 0, 'error'
        assert (torch.isnan(clustering.mol_2d_centroids_3)).sum().item() == 0, 'error'
        assert (torch.isnan(clustering.mol_3d_centroids_1)).sum().item() == 0, 'error'
        assert (torch.isnan(clustering.mol_3d_centroids_2)).sum().item() == 0, 'error'
        assert (torch.isnan(clustering.mol_3d_centroids_3)).sum().item() == 0, 'error'

        clustering.mol_2d_centroids_1 = clustering.mol_2d_centroids_1.cuda()
        clustering.mol_2d_centroids_2 = clustering.mol_2d_centroids_2.cuda()
        clustering.mol_2d_centroids_3 = clustering.mol_2d_centroids_3.cuda()
        clustering.mol_3d_centroids_1 = clustering.mol_3d_centroids_1.cuda()
        clustering.mol_3d_centroids_2 = clustering.mol_3d_centroids_2.cuda()
        clustering.mol_3d_centroids_3 = clustering.mol_3d_centroids_3.cuda()

        train_loss = train_epoch(model, optimizer, dataloader, epoch)

        cur_dir = args.save_dir + f'/Epoch_{epoch + 1}_loss_{train_loss:.5f}.bin'

        if is_master(args):
            if (epoch + 1) % 2 == 0 or epoch == 0:
                torch.save(model.module.state_dict(), cur_dir)
   

if __name__ == '__main__':
    local_rank = args.local_rank
    # local_rank = int(os.environ["LOCAL_RANK"])

    print(args)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    all_index_mapping = torch.arange(args.episode_size).share_memory_()

    # Molecular_LMDB_Dataset
    env = lmdb.open(
            args.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
    txn = env.begin()
    lmdb_env = {'txn': txn, 'keys': list(txn.cursor().iternext(values=False))}
    dataset = Molecular_LMDB_Dataset(args, lmdb_env, index_mapping=all_index_mapping, max_atom=args.train_max_atoms)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    
    dataloader = create_dataloader(dataset, args.train_batch_size, 
                                         args.num_workers, shuffle=False,
                                         pin_memory=True, drop_last=False,
                                         sampler=sampler)
    
    gconf = GraphormerConfig()
    model = GraphormerModelForPretraining(gconf, args).to(local_rank)

    if is_master(args) and args.load_ckpt != '':
        model.load_state_dict(torch.load(args.load_ckpt), strict=False)  

    # model.cl_proto_head.proto_proj_2d.weight = copy.deepcopy(model.cl_proto_head.projection_2d.weight)
    # model.cl_proto_head.proto_proj_3d.weight = copy.deepcopy(model.cl_proto_head.projection_3d.weight)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=(args.beta1, args.beta2))

    total_steps = len(dataloader) * args.train_epoch 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=int(total_steps)
    )

    clustering = Clustering(args)

    train(model, optimizer, dataloader)

    print('Done!')



