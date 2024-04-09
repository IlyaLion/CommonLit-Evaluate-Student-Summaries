import torch

import os
import gc

from dataset.dataset import CommonLitDataModule
from model.model import get_model
from criterion.criterion import get_criterion
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler
from .utils import seed_everything, train_one_epoch

def train_model(config, checkpoint_dir=None):
    seed_everything(config.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_float32_matmul_precision('medium')

    dm = CommonLitDataModule(config)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    model = get_model(config)
    model.cuda()
    
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config)

    train_steps_per_epoch = len(train_dataloader) // config.training.accumulate_grad_batches
    scheduler = get_scheduler(optimizer, config, train_steps_per_epoch)
    
    val_scores = {}
    for epoch in range(config.training.epochs):
        print(f'{epoch=}')
        train_loss, val_loss, val_score, val_columnwise_scores = train_one_epoch(config=config,
                                                                                 model=model,
                                                                                 optimizer=optimizer,
                                                                                 scheduler=scheduler, 
                                                                                 criterion=criterion, 
                                                                                 train_dataloader=train_dataloader, 
                                                                                 val_dataloader=val_dataloader)
        print(f'{train_loss=:.5f} {val_loss=:.5f}')
        print(f'{val_score=:.5f}', end=' ')
        print(f'val_columnwise_scores=({val_columnwise_scores[0]:.5f}, {val_columnwise_scores[1]:.5f})')
        val_scores[epoch] = val_score
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_filename = os.path.join(checkpoint_dir, f'fold={config.fold}-{epoch=}-{val_score=:.5f}.ckpt')
            torch.save(model.state_dict(), model_filename)

    del model
    gc.collect()
    torch.cuda.empty_cache()  

    return val_scores