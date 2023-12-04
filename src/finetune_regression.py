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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import get_cosine_schedule_with_warmup

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpts = Queue(maxsize = args.ft_max_ckpts)


def train_epoch(model, optimizer, train_dataloader, epoch, criterion):
    model.train()
    total_loss = 0
    start_time = time.time()

    for idx, batch in enumerate(train_dataloader):
        batched_data = dict()
        for key in batch.keys():
            batched_data[key] = batch[key].to(device)

        logits = model(batched_data)
        loss = criterion(logits, batched_data['label'])
        loss.backward()   

        if (idx + 1) % UPDATE_FREQ == 0 or (idx + 1) == len(train_dataloader):
            if MAX_NORM != -1:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
            optimizer.step() 
            model.zero_grad()

        total_loss += loss.item()  

        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        scheduler.step()

        if (idx + 1) % LOG_STEP == 0 or (idx + 1) == len(train_dataloader):
            print(f'Step {idx + 1}/{len(train_dataloader)}  ||  '
                  f'Train loss: {total_loss / (idx + 1):.4f}  ||  '
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

            preds = model(batched_data).cpu().squeeze(1)

            y_true.append(batched_data['label'].cpu().squeeze(1))
            y_scores.append(preds)
            
    y_true = torch.cat(y_true, dim=0).numpy()  # [n_samples]
    y_scores = torch.cat(y_scores, dim=0).numpy()

    rmse = mean_squared_error(y_true, y_scores, squared=False) # If True returns MSE value, if False returns RMSE value
    mae = mean_absolute_error(y_true, y_scores)

    return  rmse, mae

def train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, criterion):  

    best_valid_rmse = 999999
    for epoch in range(TRAIN_EPOCH):
        train_epoch(model, optimizer, train_dataloader, epoch, criterion)
        valid_rmse, valid_mae = eval(model, valid_dataloader)
        test_rmse, test_mae = eval(model, test_dataloader)
        print(f'valid rmse: {valid_rmse}  test rmse: {test_rmse}')

        cur_dir = args.ft_save_dir + f'/{args.regression_dataset}_{args.seed}_epoch{epoch+1}_AUC_{test_rmse:.3f}.bin'

        if valid_rmse < best_valid_rmse:
            if ckpts.full():
                try:
                    os.remove(ckpts.get())
                except:
                    pass
                ckpts.put(cur_dir)
                torch.save(model.state_dict(), cur_dir)
                best_valid_rmse = valid_rmse
                print(f'Best test RMSE:{test_rmse:.3f}  ->  model saved!')
            else:
                ckpts.put(cur_dir)
                torch.save(model.state_dict(), cur_dir)
                best_valid_rmse = valid_rmse
                print(f'Best test RMSE:{test_rmse:.3f}  ->  model saved!')

        print()


if __name__ == '__main__':
    SEED = args.seed
    seed_everything(SEED)
    print(f'Seed: {SEED}')
    print(f'Dataset: {args.regression_dataset}')

    if args.regression_dataset == 'esol':
        BATCH_SIZE = args.esol_batch_size
        LR = args.esol_lr
        TRAIN_EPOCH = args.esol_train_epoch
        WARMUP = args.esol_warmup
        MODEL_DROPOUT = args.esol_model_dropout
        DROPOUT = args.esol_dropout
        UPDATE_FREQ = args.esol_update_freq
        WD = args.esol_weight_decay
        MAX_NORM = args.esol_norm
        LOSS_FUNC = args.esol_loss
        CLASSES = args.esol_classes
    elif args.regression_dataset == 'freesolv':
        BATCH_SIZE = args.freesolv_batch_size
        LR = args.freesolv_lr
        TRAIN_EPOCH = args.freesolv_train_epoch
        WARMUP = args.freesolv_warmup
        MODEL_DROPOUT = args.freesolv_model_dropout
        DROPOUT = args.freesolv_dropout
        UPDATE_FREQ = args.freesolv_update_freq
        WD = args.freesolv_weight_decay
        MAX_NORM = args.freesolv_norm
        LOSS_FUNC = args.freesolv_loss
        CLASSES = args.freesolv_classes
    elif args.regression_dataset == 'lipo':
        BATCH_SIZE = args.lipo_batch_size
        LR = args.lipo_lr
        TRAIN_EPOCH = args.lipo_train_epoch
        WARMUP = args.lipo_warmup
        MODEL_DROPOUT = args.lipo_model_dropout
        DROPOUT = args.lipo_dropout
        UPDATE_FREQ = args.lipo_update_freq
        WD = args.lipo_weight_decay
        MAX_NORM = args.lipo_norm
        LOSS_FUNC = args.lipo_loss
        CLASSES = args.lipo_classes
    elif args.regression_dataset == 'malaria':
        BATCH_SIZE = args.malaria_batch_size
        LR = args.malaria_lr
        TRAIN_EPOCH = args.malaria_train_epoch
        WARMUP = args.malaria_warmup
        MODEL_DROPOUT = args.malaria_model_dropout
        DROPOUT = args.malaria_dropout
        UPDATE_FREQ = args.malaria_update_freq
        WD = args.malaria_weight_decay
        MAX_NORM = args.malaria_norm
        LOSS_FUNC = args.malaria_loss
        CLASSES = args.malaria_classes
    elif args.regression_dataset == 'cep':
        BATCH_SIZE = args.cep_batch_size
        LR = args.cep_lr
        TRAIN_EPOCH = args.cep_train_epoch
        WARMUP = args.cep_warmup
        MODEL_DROPOUT = args.cep_model_dropout
        DROPOUT = args.cep_dropout
        UPDATE_FREQ = args.cep_update_freq
        WD = args.cep_weight_decay
        MAX_NORM = args.cep_norm
        LOSS_FUNC = args.cep_loss
        CLASSES = args.cep_classes
    
    gconf = GraphormerConfig()
    gconf.hidden_dropout_prob = MODEL_DROPOUT
    gconf.attention_probs_dropout_prob = MODEL_DROPOUT
    gconf.embedding_dropout_prob = MODEL_DROPOUT

    if args.regression_dataset in ['malaria', 'cep']:
        if args.regression_dataset == 'malaria':
            df = pd.read_csv(args.molnet_path + '/malaria/malaria-processed.csv')
            train_dataset = Regression_Dataset(args, df, name='malaria', mode='train')
            valid_dataset = Regression_Dataset(args, df, name='malaria', mode='valid')
            test_dataset = Regression_Dataset(args, df, name='malaria', mode='test')
        elif args.regression_dataset == 'cep':
            df = pd.read_csv(args.molnet_path + '/cep/cep-processed.csv')
            train_dataset = Regression_Dataset(args, df, name='cep', mode='train')
            valid_dataset = Regression_Dataset(args, df, name='cep', mode='valid')
            test_dataset = Regression_Dataset(args, df, name='cep', mode='test')
    else:
        train_dataset = Molnet_Regression_Dataset(args, name=args.regression_dataset, mode='train')
        valid_dataset = Molnet_Regression_Dataset(args, name=args.regression_dataset, mode='valid')
        test_dataset = Molnet_Regression_Dataset(args, name=args.regression_dataset, mode='test')

    LOG_STEP = int(train_dataset.total_samples / BATCH_SIZE / 5)

    train_dataloader = create_dataloader(train_dataset, BATCH_SIZE, args.num_workers, shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset, batch_size=32, workers=args.num_workers, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, batch_size=32, workers=args.num_workers, shuffle=False)

    model = GraphormerModelForRegression(gconf, CLASSES, DROPOUT).to(device)
    if args.load_ckpt != '':
        model.load_state_dict(torch.load(args.load_ckpt), strict=False) 

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

    total_steps = len(train_dataloader) * TRAIN_EPOCH 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP),
        num_training_steps=int(total_steps)
    )

    if LOSS_FUNC == 'mae':
        criterion = nn.L1Loss()
    elif LOSS_FUNC == 'mse':
        criterion = nn.MSELoss()
    elif LOSS_FUNC == 'smooth_l1':
        criterion = nn.SmoothL1Loss()

    train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, criterion)

    print('Done!')



