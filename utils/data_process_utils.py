# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 1:53 下午
# @Author  : lizhen
# @FileName: data_process_utils.py
# @Description:
import csv
from collections import defaultdict
from copy import deepcopy

import torch
from harvesttext import HarvestText
from transformers import AlbertConfig, AlbertModel, BertTokenizer, BertConfig, BertModel, BertForPreTraining, \
    load_tf_weights_in_bert, LineByLineTextDataset


# from seqeval.metrics import


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


def convert_tag_ids_tags(tag_id):
    """

    :param tag_ids:[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    :return: [[0,2],[3,4],[16,34],[38,39]]
    """
    tag = []
    for i in tag_id:
        if i == 0:
            tag.append('O')
        elif i == 1:
            tag.append('B-ner')
        elif i == 2:
            tag.append('I-ner')
    return tag


def get_raw_instance(document, max_sequence_length):  # 新增的方法
    """
    获取初步的训练实例，将整段按照max_sequence_length切分成多个部分,并以多个处理好的实例的形式返回。
    :param document: 一整段
    :param max_sequence_length:
    :return: a list. each element is a sequence of text
    """
    max_sequence_length_allowed = max_sequence_length - 2
    document = [seq for seq in document if len(
        seq) < max_sequence_length_allowed]
    sizes = [len(seq) for seq in document]

    result_list = []
    curr_seq = []  # 当前处理的序列
    sz_idx = 0
    while sz_idx < len(sizes):
        # 当前句子加上新的句子，如果长度小于最大限制，则合并当前句子和新句子；否则即超过了最大限制，那么做为一个新的序列加到目标列表中
        # or len(curr_seq)==0:
        if len(curr_seq) + sizes[sz_idx] <= max_sequence_length_allowed:
            curr_seq += document[sz_idx]
            sz_idx += 1
        else:
            result_list.append(curr_seq)
            curr_seq = []
    # 对最后一个序列进行处理，如果太短的话，丢弃掉。
    if len(curr_seq) > max_sequence_length_allowed / 2:  # /2
        result_list.append(curr_seq)

    # # 计算总共可以得到多少份
    # num_instance=int(len(big_list)/max_sequence_length_allowed)+1
    # print("num_instance:",num_instance)
    # # 切分成多份，添加到列表中
    # result_list=[]
    # for j in range(num_instance):
    #     index=j*max_sequence_length_allowed
    #     end_index=index+max_sequence_length_allowed if j!=num_instance-1 else -1
    #     result_list.append(big_list[index:end_index])
    return result_list


def create_instances_from_document(

        # 目标按照RoBERTa的思路，使用DOC-SENTENCES，并会去掉NSP任务: 从一个文档中连续的获得文本，直到达到最大长度。如果是从下一个文档中获得，那么加上一个分隔符
        #  document即一整段话，包含多个句子。每个句子叫做segment.
        # 给定一个document即一整段话，生成一些instance.
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.

    # target_seq_length = max_num_tokens
    # if rng.random() < short_seq_prob:
    #    target_seq_length = rng.randint(2, max_num_tokens)

    instances = []
    # document即一整段话，包含多个句子。每个句子叫做segment.
    raw_text_list_list = get_raw_instance(document, max_seq_length)
    for j, raw_text_list in enumerate(raw_text_list_list):
        ####################################################################################################################
        # 结合分词的中文的whole mask设置即在需要的地方加上“##”
        raw_text_list = get_new_segment(raw_text_list)
        # 1、设置token, segment_ids
        is_random_next = True  # this will not be used, so it's value doesn't matter
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in raw_text_list:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        ################################################################################################################
        # 2、调用原有的方法
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def generate_multiline(line: str, max_len=510):
    line = line.lstrip('，').lstrip(',')

    if len(line) < max_len:
        return [line]

    ht = HarvestText()
    sentences = ht.cut_sentences(line, deduplicate=True)
    result = []

    line_len = 0
    s = ''
    remained = False
    while sentences:
        sentence = sentences.pop(0).strip()
        sent_len = len(sentence)
        if sent_len > 510:
            if s:
                result.append(s)
                s = ''
                line_len = 0
            result.append(sentence[:510])
            continue
        if line_len + sent_len <= 510:
            s += sentence
            line_len += sent_len
            remained = True
        else:
            result.append(s)
            line_len = 0
            s = sentence
            remained = False
    if remained:
        result.append(s)

    return result
