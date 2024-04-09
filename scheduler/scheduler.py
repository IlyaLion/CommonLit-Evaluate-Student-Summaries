from torch.optim.lr_scheduler import OneCycleLR

def get_scheduler(optimizer, config, steps_per_epoch):
    scheduler = OneCycleLR(optimizer, 
                           max_lr=[g['lr'] for g in optimizer.param_groups],
                           epochs=config.training.epochs,
                           steps_per_epoch=steps_per_epoch,
                           anneal_strategy=config.scheduler.anneal_strategy,
                           pct_start=config.scheduler.pct_start,
                           div_factor=config.scheduler.div_factor,
                           final_div_factor=config.scheduler.final_div_factor)

    return scheduler