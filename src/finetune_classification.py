'''
run this script three times with seed 1, 2, 3 in args
'''
from config import args
from queue import Queue
from dataset import *
from utils import *
from models import *
import numpy as np
import os
import time
from tqdm import tqdm
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpts = Queue(maxsize = args.ft_max_ckpts)
criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train_epoch(model, optimizer, train_dataloader, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    total_acc = 0

    for idx, batch in enumerate(train_dataloader):
        batched_data = dict()
        for key in batch.keys():
            batched_data[key] = batch[key].to(device)

        logits = model(batched_data)
        is_valid = batched_data['w']**2 > 0
        loss_mat = criterion(logits, batched_data['label'])
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()   

        if (idx + 1) % UPDATE_FREQ == 0 or (idx + 1) == len(train_dataloader):
            optimizer.step() 
            model.zero_grad()

        preds = torch.where(torch.sigmoid(logits.detach().cpu()) > 0.5, 1, 0).reshape(-1)
        acc = batched_data['label'].reshape(-1).detach().cpu().eq(preds).sum() / preds.shape[0]

        total_acc += acc
        total_loss += loss.item()  

        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

        scheduler.step()

        if (idx + 1) % LOG_STEP == 0 or (idx + 1) == len(train_dataloader):
            print(f'Step {idx + 1}/{len(train_dataloader)}  ||  '
                  f'Train loss: {total_loss / (idx + 1):.4f}  ||  '
                  f'Current acc: {total_acc / (idx + 1):.4f}  ||  '
                  f'Time: {((time.time() - start_time) / 60.0):.1f}min  ||  '
                  f'lr: {cur_lr:g}')

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(' ')
    print(f'Epoch {epoch + 1} time: {epoch_mins}m {epoch_secs}s')


def eval(model, valid_dataloader):
    model.eval()
    y_true = []
    y_scores = []
    ws = []

    with torch.no_grad():
        for idx, batch in enumerate(valid_dataloader):
            batched_data = dict()
            for key in batch.keys():
                batched_data[key] = batch[key].to(device)

            preds = model(batched_data).cpu()
            # preds = torch.sigmoid(preds)

            y_true.append(batched_data['label'].cpu())
            y_scores.append(preds)
            ws.append(batched_data['w'].cpu())
            
    y_true = torch.cat(y_true, dim = 0).numpy()  # [n_samples, classes]
    y_scores = torch.cat(y_scores, dim = 0).numpy()
    ws = torch.cat(ws, dim = 0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        is_valid = ws[:, i]**2 > 0
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[is_valid, i] == 1) > 0 and np.sum(y_true[is_valid, i] == 0) > 0:
            # is_valid = ws[:, i]**2 > 0
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    avg_roc_auc = sum(roc_list)/len(roc_list)

    return  avg_roc_auc

def train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader):  

    best_valid_AUC = 0
    for epoch in range(TRAIN_EPOCH):
        train_epoch(model, optimizer, train_dataloader, epoch)
        valid_AUC = eval(model, valid_dataloader)
        test_AUC = eval(model, test_dataloader)
        print(f'valid auc: {valid_AUC}  test_auc: {test_AUC}')

        cur_dir = args.ft_save_dir + f'/{args.classification_dataset}_{args.seed}_epoch{epoch+1}_AUC_{test_AUC:.3f}.bin'

        if valid_AUC > best_valid_AUC:
            if ckpts.full():
                try:
                    os.remove(ckpts.get())
                except:
                    pass
                ckpts.put(cur_dir)
                torch.save(model.state_dict(), cur_dir)
                best_valid_AUC = valid_AUC
                print(f'Best test ROC-AUC:{test_AUC:.3f}  ->  model saved!')
            else:
                ckpts.put(cur_dir)
                torch.save(model.state_dict(), cur_dir)
                best_valid_AUC = valid_AUC
                print(f'Best test ROC-AUC:{test_AUC:.3f}  ->  model saved!')

        print()


if __name__ == '__main__':
    SEED = args.seed
    seed_everything(SEED)
    print(f'Seed: {SEED}')
    print(f'Dataset: {args.classification_dataset}')

    gconf = GraphormerConfig()

    if args.classification_dataset == 'bbbp':
        BATCH_SIZE = args.bbbp_batch_size
        LR = args.bbbp_lr
        TRAIN_EPOCH = args.bbbp_train_epoch
        WARMUP = args.bbbp_warmup
        DROPOUT = args.bbbp_dropout
        UPDATE_FREQ = args.bbbp_update_freq
        WD = args.bbbp_weight_decay
        CLASSES = args.bbbp_classes
    elif args.classification_dataset == 'bace':
        BATCH_SIZE = args.bace_batch_size
        LR = args.bace_lr
        TRAIN_EPOCH = args.bace_train_epoch
        WARMUP = args.bace_warmup
        DROPOUT = args.bace_dropout
        UPDATE_FREQ = args.bace_update_freq
        WD = args.bace_weight_decay
        CLASSES = args.bace_classes
    elif args.classification_dataset == 'clintox':
        BATCH_SIZE = args.clintox_batch_size
        LR = args.clintox_lr
        TRAIN_EPOCH = args.clintox_train_epoch
        WARMUP = args.clintox_warmup
        DROPOUT = args.clintox_dropout
        UPDATE_FREQ = args.clintox_update_freq
        WD = args.clintox_weight_decay
        CLASSES = args.clintox_classes
    elif args.classification_dataset == 'tox21':
        BATCH_SIZE = args.tox21_batch_size
        LR = args.tox21_lr
        TRAIN_EPOCH = args.tox21_train_epoch
        WARMUP = args.tox21_warmup
        DROPOUT = args.tox21_dropout
        UPDATE_FREQ = args.tox21_update_freq
        WD = args.tox21_weight_decay
        CLASSES = args.tox21_classes
    elif args.classification_dataset == 'toxcast':
        BATCH_SIZE = args.toxcast_batch_size
        LR = args.toxcast_lr
        TRAIN_EPOCH = args.toxcast_train_epoch
        WARMUP = args.toxcast_warmup
        DROPOUT = args.toxcast_dropout
        UPDATE_FREQ = args.toxcast_update_freq
        WD = args.toxcast_weight_decay
        CLASSES = args.toxcast_classes
    elif args.classification_dataset == 'sider':
        BATCH_SIZE = args.sider_batch_size
        LR = args.sider_lr
        TRAIN_EPOCH = args.sider_train_epoch
        WARMUP = args.sider_warmup
        DROPOUT = args.sider_dropout
        UPDATE_FREQ = args.sider_update_freq
        WD = args.sider_weight_decay
        CLASSES = args.sider_classes

    train_dataset = Molnet_Classification_Dataset(args, name=args.classification_dataset, mode='train')
    valid_dataset = Molnet_Classification_Dataset(args, name=args.classification_dataset, mode='valid')
    test_dataset = Molnet_Classification_Dataset(args, name=args.classification_dataset, mode='test')

    LOG_STEP = int(train_dataset.total_samples / BATCH_SIZE / 5)

    train_dataloader = create_dataloader(train_dataset, BATCH_SIZE, args.num_workers, shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset, batch_size=14, workers=args.num_workers, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, batch_size=14, workers=args.num_workers, shuffle=False)

    model = GraphormerModelForClassification(gconf, args, CLASSES, DROPOUT).to(device)
    if args.load_ckpt != '':
        model.load_state_dict(torch.load(args.load_ckpt), strict=False) 

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

    total_steps = len(train_dataloader) * TRAIN_EPOCH 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP),
        num_training_steps=int(total_steps)
    )

    train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)

    print('Done!')



