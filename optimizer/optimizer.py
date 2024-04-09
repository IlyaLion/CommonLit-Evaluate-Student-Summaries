from .parameters import get_optimizer_one_lr_params, get_optimizer_two_lrs_params
from torch.optim import AdamW


def get_optimizer(model, config):
    if config.optimizer.type == 'one_lr':
        optimizer_parameters = get_optimizer_one_lr_params(model,
                                                           lr=config.optimizer.encoder_lr,
                                                           weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.type == 'two_lrs':
        optimizer_parameters = get_optimizer_two_lrs_params(model,
                                                            encoder_lr=config.optimizer.encoder_lr,
                                                            decoder_lr=config.optimizer.decoder_lr,
                                                            weight_decay=config.optimizer.weight_decay)
    else:
        raise ValueError(f'Invalid optimizer type: {config.optimizer.type}')
    optimizer = AdamW(optimizer_parameters,
                      lr=config.optimizer.encoder_lr,
                      eps=config.optimizer.eps,
                      betas=config.optimizer.betas)

    return optimizer