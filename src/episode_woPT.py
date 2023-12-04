from config import args
from queue import Queue
from dataset import *
from distributed import *
from utils import epoch_time, seed_everything, feature_extraction
import numpy as np
import os
import time
from tqdm import tqdm
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda import amp
from transformers import get_cosine_schedule_with_warmup
from models import GraphormerConfig, GraphormerModelForCLandIM
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

    CL_acc_accum = 0
    im2d_acc_accum = 0

    for idx, batch in enumerate(train_dataloader):
        batched_data = dict()
        for key in batch.keys():
            batched_data[key] = batch[key].cuda()

        if args.use_amp and epoch < args.stop_amp_epoch:
            with amp.autocast():
                loss_CL, loss_infomotif_2d, loss_denoise, CL_acc, im2d_acc = model(batched_data)
                loss = loss_CL + loss_infomotif_2d + loss_denoise
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss_CL, loss_infomotif_2d, loss_denoise, CL_acc, im2d_acc = model(batched_data)
            loss = loss_CL + loss_infomotif_2d + loss_denoise
            loss.backward()    
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            optimizer.step() 

        total_loss += loss.item()  
        total_loss_CL += loss_CL.item() 
        total_loss_infomotif_2d += loss_infomotif_2d.item() 
        total_loss_denoise += loss_denoise.item()

        CL_acc_accum += CL_acc
        im2d_acc_accum += im2d_acc

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
                    f'CL acc: {(CL_acc_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'im acc: {(im2d_acc_accum / (idx + 1)) * 100.0:.3f}% | '
                    f'Time: {((time.time() - start_time) / 60.0):.1f}min | '
                    f'lr: {cur_lr:g}')

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if is_master(args):
        print(' ')
        print(f'Epoch {epoch + 1} time: {epoch_mins}m {epoch_secs}s train loss: {total_loss / len(train_dataloader):.4f}')
        print(f'CL acc: {(CL_acc_accum / len(train_dataloader)) * 100.0:.6f}%  im acc: {(im2d_acc_accum / len(train_dataloader)) * 100.0:.6f}%  '
              f'denoise loss: {total_loss_denoise / len(train_dataloader):.6f} ') 
        print(' ')

    return total_loss / len(train_dataloader)


def eval(model, valid_dataloader, epoch):
    model.eval()

    total_loss = 0
    total_loss_CL = 0
    total_loss_infomotif_2d = 0
    total_loss_denoise = 0

    CL_acc_accum = 0
    im2d_acc_accum = 0

    with torch.no_grad():
        for idx, batch in enumerate(valid_dataloader):
            batched_data = dict()
            for key in batch.keys():
                batched_data[key] = batch[key].to(local_rank)

            loss_CL, loss_infomotif_2d, loss_denoise, CL_acc, im2d_acc = model(batched_data)
            loss = loss_CL + loss_infomotif_2d + loss_denoise 

            total_loss += loss.item()  
            total_loss_CL += loss_CL.item() 
            total_loss_infomotif_2d += loss_infomotif_2d.item() 
            total_loss_denoise += loss_denoise.item() 

            CL_acc_accum += CL_acc
            im2d_acc_accum += im2d_acc
    
    total_loss = total_loss / len(valid_dataloader)
    total_loss_CL = total_loss_CL / len(valid_dataloader)
    total_loss_infomotif_2d = total_loss_infomotif_2d / len(valid_dataloader)
    total_loss_denoise = total_loss_denoise / len(valid_dataloader)

    CL_acc_accum = CL_acc_accum / len(valid_dataloader)
    im2d_acc_accum = im2d_acc_accum / len(valid_dataloader)

    if is_master(args):
        print(f'total valid loss: {total_loss:.2f}  '
              f'valid CL loss:  {total_loss_CL:.2f}  '
              f'im loss:  {total_loss_infomotif_2d:.2f}  '
              f'denoise loss:  {total_loss_denoise:.6f}  '
              f'CL acc:  {CL_acc_accum * 100.0:.5f}%  '
              f'im acc:  {im2d_acc_accum * 100.0:.5f}%  ')

    return total_loss


def train(model, optimizer, train_dataloader, valid_dataloader=None):  

    best_loss = 99999
    for epoch in range(args.train_epoch):

        index_mapping_train[:] = torch.from_numpy(np.random.choice(args.dataset_size, args.episode_size, replace=False))
        index_mapping_valid[:] = torch.from_numpy(np.random.choice(args.dataset_size, int(args.episode_size * args.valid_ratio), replace=False))

        train_loss = train_epoch(model, optimizer, train_dataloader, epoch)

        if is_master(args) and epoch == 0:
            cur_dir = args.save_dir + f'/Epoch_{epoch + 1}.bin'
            torch.save(model.module.state_dict(), cur_dir)

        if (epoch + 1) % args.valid_interval == 0:
            valid_loss = eval(model, valid_dataloader, epoch)

            cur_dir = args.save_dir + f'/Epoch_{epoch + 1}_loss_{valid_loss:.3f}.bin'

            if is_master(args):
                if (epoch + 1) % 5 == 0:
                    torch.save(model.module.state_dict(), cur_dir)
                    
                if valid_loss < best_loss:
                    if ckpts.full():
                        try:
                            os.remove(ckpts.get())
                        except:
                            pass
                        ckpts.put(cur_dir)
                        torch.save(model.module.state_dict(), cur_dir)
                        best_loss = valid_loss
                        print(f'Best loss:{best_loss:.3f}  ->  model saved!')
                    else:
                        ckpts.put(cur_dir)
                        torch.save(model.module.state_dict(), cur_dir)
                        best_loss = valid_loss
                        print(f'Best loss:{best_loss:.3f}  ->  model saved!')
                print()
   

if __name__ == '__main__':
    local_rank = args.local_rank

    print(args)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    index_mapping_train = torch.arange(args.episode_size).share_memory_()
    index_mapping_valid = torch.arange(int(args.episode_size * args.valid_ratio)).share_memory_()

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
    train_dataset = Molecular_LMDB_Dataset(args, lmdb_env, index_mapping=index_mapping_train, max_atom=args.train_max_atoms)
    valid_dataset = Molecular_LMDB_Dataset(args, lmdb_env, index_mapping=index_mapping_valid, max_atom=args.valid_max_atoms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    
    train_dataloader = create_dataloader(train_dataset, args.train_batch_size, 
                                         args.num_workers, shuffle=False,
                                         pin_memory=True, drop_last=False,
                                         sampler=train_sampler)

    valid_dataloader = create_dataloader(valid_dataset, args.valid_batch_size, 
                                         args.num_workers, shuffle=False,
                                         pin_memory=True, drop_last=False,
                                         sampler=valid_sampler)
    
    gconf = GraphormerConfig()
    model = GraphormerModelForCLandIM(gconf, args).to(local_rank)

    if is_master(args) and args.load_ckpt != '':
        model.load_state_dict(torch.load(args.load_ckpt), strict=False)  
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=(args.beta1, args.beta2))

    total_steps = len(train_dataloader) * args.train_epoch * 1.05
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=int(total_steps)
    )

    train(model, optimizer, train_dataloader, valid_dataloader)

    print('Done!')



