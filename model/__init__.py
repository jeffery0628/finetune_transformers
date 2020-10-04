# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 2:52 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch
import transformers

import model.loss as module_loss
import model.metric as module_metric
import model.models as module_models

__all__ = ["makeModel", "makeLoss", "makeMetrics", "makeOptimizer", "makeLrSchedule"]


def makeModel(config):
    return config.init_obj('model', module_models, )


def makeLoss(config):
    return [getattr(module_loss, crit) for crit in config['loss']]


def makeMetrics(config):
    return [getattr(module_metric, met) for met in config['metrics']]


def makeOptimizer(config, model):
    bert_params = [*filter(lambda p: p.requires_grad, model.transformer_model.parameters())]
    # U_params = [*filter(lambda p: p.requires_grad, model.U.parameters())]
    # final_params = [*filter(lambda p:p.requires_grad,model.final.parameters())]

    optimizer = config.init_obj('optimizer', torch.optim, [
        {'params': bert_params, 'lr': 1e-4, "weight_decay": 0.},
        # {'params': U_params, 'lr': 5e-4, "weight_decay": 0.85},
        # {'params': final_params, 'lr': 3e-3, "weight_decay": 0.85},
    ])

    return optimizer


def makeLrSchedule(config, optimizer, train_dataloader):
    # lr_scheduler = config.init_obj('lr_scheduler', optimization.lr_scheduler, optimizer)
    lr_scheduler = config.init_obj('lr_scheduler', transformers.optimization, optimizer,
                                   num_training_steps=int(len(train_dataloader.dataset) / train_dataloader.batch_size))
    return lr_scheduler
