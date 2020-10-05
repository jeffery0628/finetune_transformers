# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 4:46 下午
# @Author  : jeffery
# @FileName: train.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:

import random

import numpy as np
import torch
import yaml

from data_process import makeDataLoader
from model import makeModel, makeLoss, makeMetrics, makeOptimizer, makeLrSchedule
from trainer import Trainer
from utils import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    train_dataloader, valid_dataloader = makeDataLoader(config)

    model = makeModel(config)
    logger.info(model)

    # criterion = makeLoss(config)
    # metrics = makeMetrics(config)

    optimizer = makeOptimizer(config, model)
    lr_scheduler = makeLrSchedule(config, optimizer, train_dataloader)

    trainer = Trainer(model, None, None, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      test_data_loader=None,
                      lr_scheduler=lr_scheduler)
    trainer.train()

def run(config_fname):
    with open(config_fname, 'r', encoding='utf8') as f:
        config_params = yaml.load(f, Loader=yaml.Loader)
        config_params['config_file_name'] = config_fname

    config = ConfigParser.from_args(config_params)
    main(config)


if __name__ == '__main__':
    run('configs/config.yml')
