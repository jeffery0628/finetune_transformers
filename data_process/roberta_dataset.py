# -*- coding: utf-8 -*-
# @Time    : 2020/9/2 2:38 下午
# @Author  : jeffery
# @FileName: sentence_transformers_dataset.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import os
import random
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, TextDataset, Trainer, DataCollatorForLanguageModeling, RobertaTokenizer, \
    LineByLineTextDataset

from base import BaseDataSet
from utils import printable_text, convert_to_unicode, FullTokenizer

print(os.getcwd())
"""Create masked LM/next sentence masked_lm examples for BERT."""
