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
from transformers import AlbertModel, AlbertConfig

from base import BaseModel


class AttentionCls(BaseModel):

    def __init__(self, pretrained_dir, transformer_train, dropout, class_num, init_fc_file):
        super(AttentionCls, self).__init__()
        config = AlbertConfig.from_pretrained(pretrained_dir)

        self.transformer_model = AlbertModel.from_pretrained(os.path.join(pretrained_dir, 'pytorch_model.bin'),
                                                             config=config)
        # 是否对bert进行训练
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = transformer_train

        self.dropout = dropout

        # self.U = nn.Linear(self.transformer_model.config.to_dict()['hidden_size'], class_num)

        self.final = nn.Linear(self.transformer_model.config.to_dict()['hidden_size'], class_num)
        # self.__init_fc(init_fc_file)

    def __init_fc(self, vector_file):
        with open(vector_file, 'rb') as f:
            code_vector = pickle.load(f)
            code_tensor = torch.FloatTensor(np.array(code_vector)).squeeze()
            # self.final.weight.data = code_tensor.clone()

            self.U.weight.data = code_tensor.clone()

    def forward(self, text_ids, masks):
        seq_out, seq_cls = self.transformer_model(text_ids, attention_mask=masks)
        # bert_out = self.dropout(bert_out)

        # # self.U.weight : [num_type,emb_size]  bert_out : [bs,seq_len,emb_size]   ---> alpha : [bs,num_type,seq_len]
        # alpha = F.softmax(self.U.weight.matmul(seq_out.transpose(1, 2)), dim=2)
        # # document representations are weighted sums using the attention. Can compute all at once as a matmul
        #
        # # m: [bs, num_type,emb_size]
        # m = alpha.matmul(seq_out)
        # # final layer classification
        # # self.final : [emb_size,num_type]   output : [bs,num_type]
        # output = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        output = self.final(seq_cls)
        return output


class SentenceTransformers(BaseModel):

    def __init__(self, pretrained_dir, transformer_train):
        super(SentenceTransformers, self).__init__()
        config = AlbertConfig.from_pretrained(pretrained_dir)

        self.transformer_model = AlbertModel.from_pretrained(os.path.join(pretrained_dir, 'pytorch_model.bin'),
                                                             config=config)
        # 是否对bert进行训练
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = transformer_train

    def forward(self, text_ids, masks):
        seq_out, seq_cls = self.transformer_model(text_ids, attention_mask=masks)
        # mean pooling
        output_vectors = []
        input_mask_expanded = masks.unsqueeze(-1).expand(seq_out.size()).float()
        sum_embeddings = torch.sum(seq_out * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector
