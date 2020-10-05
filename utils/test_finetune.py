# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 9:01 上午
# @Author  : jeffery
# @FileName: test_finetune.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import os

from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, \
    LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, RobertaTokenizerFast,PreTrainedModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def finetune_model(transformers_model_name: str, corpus_file_path: str):
    config = AutoConfig.from_pretrained(transformers_model_name, force_download=False,
                                        cache_dir='../data/download_transformer_models')

    tokenizer = AutoTokenizer.from_pretrained(transformers_model_name, force_download=False,
                                              cache_dir='../data/download_transformer_models')
    # tokenizer = RobertaTokenizerFast.from_pretrained(transformers_model_name,force_download=False,cache_dir='../data/download_transformer_models')

    model = AutoModelForMaskedLM.from_pretrained(transformers_model_name, config=config, force_download=False,
                                                 cache_dir='../data/download_transformer_models')
    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=corpus_file_path, block_size=512)
    train_set, valid_set = train_test_split(dataset, test_size=0.25, random_state=32)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="../data/finetune_transformer_models/",
        logging_dir='../saved/finetune_logging',
        logging_steps=500,
        overwrite_output_dir=True,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=32,
        max_grad_norm=5.0,
        save_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=32,
        evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        do_predict=False

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=valid_set,

    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainer.train()


if __name__ == '__main__':
    transformers_model_name = "hfl/chinese-bert-wwm"
    corpus_file_path = "../data/corpus/all_corpus.txt"
    finetune_model(transformers_model_name, corpus_file_path)
