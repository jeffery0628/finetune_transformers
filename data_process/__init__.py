# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:08 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
# import data_process.mimic_version as module_data_process
from torch.utils.data import dataloader as module_dataloader

import data_process.language_model_dataprocess as module_data_process


def makeDataLoader(config):
    # setup data_set, data_process instances
    dataset = config.init_obj('data_set', module_data_process)

    train_dataloader = config.init_obj('train_loader', module_dataloader, dataset.train_feature,
                                       collate_fn=dataset.collate_fn)

    valid_dataloader = config.init_obj('valid_loader', module_dataloader, dataset.valid_feature,
                                       collate_fn=dataset.collate_fn)

    return train_dataloader, valid_dataloader
