# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 1:53 下午
# @Author  : lizhen
# @FileName: data_process_utils.py
# @Description:
import csv
from collections import defaultdict
from copy import deepcopy

import torch
from transformers import AlbertConfig, AlbertModel, BertTokenizer, BertConfig, BertModel, BertForPreTraining, \
    load_tf_weights_in_bert, LineByLineTextDataset
from pathlib import Path
import json


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


def brat_to_docanno(brat_dir:Path,out_file:Path):
    writer = out_file.open('w')
    ann_files = brat_dir.glob('*.ann')
    text_files = brat_dir.glob('*.txt')
    ann_files = sorted(ann_files)
    text_files = sorted(text_files)
    for ann_file,text_file in zip(ann_files,text_files):
        tid = ann_file.stem
        text = text_file.open('r').read().rstrip()
        labels = []
        with ann_file.open('r') as ann_f:
            for line in ann_f:
                ent_id,entity_type,start_pos,end_pos,entity_text = line.split()
                assert text[int(start_pos):int(end_pos)] == entity_text,'text not same'
                labels.append({
                    "start_pos":int(start_pos),
                    "end_pos":int(end_pos),
                    "entity_type":entity_type,
                    "entity_text":entity_text
                })
        item = {
            "tid":tid,
            "text":text,
            "labels":labels
        }
        writer.write(json.dumps(item,ensure_ascii=False)+'\n')







if __name__ == '__main__':
    brat_dir = Path('../data/chinese_medical_ner/train')
    out_file = Path('../data/chinese_medical_ner/train.jsonl')
    brat_to_docanno(brat_dir,out_file)