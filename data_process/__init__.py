# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:08 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
# import data_process.mimic_version as module_data_process
from torch.utils.data import dataloader as module_dataloader

import data_process.roberta_dataset as module_data_process


def makeDataLoader(config):
    # setup data_set, data_process instances
    train_set = config.init_obj('train_set', module_data_process)
    valid_set = config.init_obj('valid_set', module_data_process)
    test_set = config.init_obj('test_set', module_data_process)
    # train_set = valid_set
    print(len(train_set), len(valid_set), len(test_set))
    train_dataloader = config.init_obj('train_loader', module_dataloader, train_set.features,
                                       collate_fn=train_set.collate_fn)

    valid_dataloader = config.init_obj('valid_loader', module_dataloader, valid_set.features,
                                       collate_fn=valid_set.collate_fn)
    test_dataloader = config.init_obj('test_loader', module_dataloader, test_set.features,
                                      collate_fn=valid_set.collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader
