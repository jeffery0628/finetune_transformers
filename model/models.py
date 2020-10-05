# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 2:52 下午
# @Author  : jeffery
# @FileName: models.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AlbertModel, AlbertConfig,AlbertForMaskedLM

from base import BaseModel


class LMNER(BaseModel):
    def __init__(self,transformer_model,is_train):
        super(LMNER,self).__init__()
        config = AlbertConfig.from_pretrained(transformer_model)
        self.transformer_model = AlbertForMaskedLM.from_pretrained(transformer_model,config=config)
        # 是否对bert进行训练
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train


    def forward(self, input_ids, labels_ids, masked_pos):
        outputs = self.transformer_model(input_ids, masked_lm_labels=labels_ids)
        loss, prediction_scores = outputs[:2]
        return loss