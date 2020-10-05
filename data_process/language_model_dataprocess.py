# -*- coding: utf-8 -*-
# @Time    : 2020/9/2 2:38 下午
# @Author  : jeffery
# @FileName: sentence_transformers_dataset.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import random
from pathlib import Path
from typing import Optional, List
from collections import OrderedDict
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from base import BaseDataSet
import json
from dataclasses import dataclass
import re
from transformers import BertForNextSentencePrediction, DataCollatorForSOP,Trainer
import numpy as np

@dataclass
class LMInputExample:
    tid: Optional[str]
    text: str
    labels: List

    def __post_init__(self):
        """
        1. 分句
        2. 把分句后的ner位置对齐
        """
        self.labels = sorted(self.labels, key=lambda x: x['start_pos'])
        self.sentence_dict = OrderedDict()
        sentences = re.findall('.*?。', self.text)
        sentence_start = 0
        for idx, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            sentence_labels = []

            while self.labels and int(self.labels[0]['start_pos']) - sentence_start >= 0 and int(
                    self.labels[0]['end_pos']) - sentence_start <= sentence_len:
                start_pos = int(self.labels[0]['start_pos']) - sentence_start
                end_pos = int(self.labels[0]['end_pos']) - sentence_start
                assert sentence[start_pos:end_pos] == self.labels[0]['entity_text'], 'entity text is not same'
                label = self.labels.pop(0)
                label['start_pos'] = start_pos
                label['end_pos'] = end_pos
                sentence_labels.append(label)

            self.sentence_dict[str(idx)] = {
                'sentence_text': sentence,
                'sentence_labels': sentence_labels
            }
            sentence_start += sentence_len


@dataclass
class LMInputFeatuer:
    tid: Optional[str]  # document id
    sid: Optional[str]  # sentence id
    input_ids: List[int]
    label_ids: List[int]
    masked_pos: List[int]
    def __post_init__(self):
        self.sent_len = len(self.input_ids)


class LMDataset(BaseDataSet):
    def __init__(self, data_dir: str, file_name: str, shuffle: bool, data_mode: str, valid_size: float,
                 transformer_model: str, overwrite_cache: bool):
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.shuffle = shuffle
        self.data_mode = data_mode
        self.valid_size = valid_size
        self.feature_cache_file = self.data_dir / '.cache' / '{}.cache'.format(file_name.split('.')[0])
        super(LMDataset, self).__init__(transformer_model, overwrite_cache)
        self.train_feature, self.valid_feature = train_test_split(self.features, test_size=0.25)

    def read_examples_from_file(self):
        input_file = self.data_dir / self.file_name
        with input_file.open('r') as f:
            for line in tqdm(f):
                json_line = json.loads(line)
                yield LMInputExample(json_line['tid'], json_line['text'], json_line['labels'])

    def convert_examples_to_features(self):
        features = []
        for exam in self.read_examples_from_file():
            for sentence_id, sentence in exam.sentence_dict.items():
                # 观察数据，实体数量较多。而mask住全部实体会丢失句子过多语义信息。因此对于每个实体有60%的概率被mask住。
                label_text = sentence['sentence_text']
                masked_text = list(label_text)
                masked_pos = [0] * len(masked_text)
                for label in sentence['sentence_labels']:
                    random_float = random.random()
                    if random_float < 0.5:
                        start_pos = int(label['start_pos'])
                        end_pos = int(label['end_pos'])
                        masked_text[start_pos:end_pos] = [self.tokenizer.mask_token] * (end_pos - start_pos)
                        masked_pos[start_pos:end_pos] = [1] * (end_pos - start_pos)
                input_ids = self.tokenizer.encode(list(label_text), add_special_tokens=True)
                label_ids = self.tokenizer.encode(masked_text, add_special_tokens=True)
                masked_pos = [0] + masked_pos + [0]
                assert len(input_ids) == len(label_ids), 'input_ids length is not equal to label_ids length'
                features.append(LMInputFeatuer(tid=exam.tid, sid=sentence_id, input_ids=input_ids, label_ids=label_ids,
                                               masked_pos=masked_pos))
        return features

    def collate_fn(self,datas):
        max_sent_len = max([data.sent_len for data in datas])

        input_ids = []
        labels_ids = []
        masked_pos = []
        for data in datas:
            input_ids.append(data.input_ids+[self.tokenizer.pad_token_id]*(max_sent_len - data.sent_len))
            labels_ids.append(data.label_ids + [self.tokenizer.pad_token_id] * (max_sent_len-data.sent_len))
            masked_pos.append(data.masked_pos + [0] * (max_sent_len-data.sent_len))

        input_ids = torch.LongTensor(np.asarray(input_ids))
        labels_ids = torch.LongTensor(np.asarray(labels_ids))
        masked_pos = torch.LongTensor(np.asarray(masked_pos))
        return input_ids,labels_ids,masked_pos
