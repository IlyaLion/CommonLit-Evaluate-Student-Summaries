import torch

import os
import random
from tqdm import tqdm
import numpy as np

from dataset.collators import collate
from criterion.score import get_score

    
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def train_one_epoch(config, model, optimizer, scheduler, criterion, train_dataloader, val_dataloader):
    match config.training.precision:
        case 16:
            amp_enabled = True
            amp_dtype = torch.float16
        case 'bf16':
            amp_enabled = True
            amp_dtype = torch.bfloat16
        case 32:
            amp_enabled = False
            amp_dtype = torch.float32
        case _:
            raise ValueError(f'Incorrect precision value: {config.training.precision}')
        
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    train_losses = AverageMeter()
    
    for step, (inputs, labels) in enumerate(tqdm(train_dataloader, desc='training')):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        labels = labels.cuda()
        
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        
        if config.training.accumulate_grad_batches > 1:
            loss = loss / config.training.accumulate_grad_batches
        
        batch_size = labels.size(0)   
        train_losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        
        if (step + 1) % config.training.accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
       
            
    val_losses = AverageMeter()
    model.eval()
    val_labels, val_predictions = [], []       
    for step, (inputs, labels) in enumerate(tqdm(val_dataloader, desc='validation')):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        labels = labels.cuda()

        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype), torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        if config.training.accumulate_grad_batches > 1:
            loss = loss / config.training.accumulate_grad_batches
    
        val_losses.update(loss.item(), batch_size)
        for i in range(batch_size):
            val_predictions.append(y_preds[i].float().cpu().numpy())
            val_labels.append(labels[i].cpu().numpy())
        
    val_score, val_columnwise_scores = get_score(np.array(val_labels), np.array(val_predictions))
    return train_losses.avg, val_losses.avg, val_score, val_columnwise_scores