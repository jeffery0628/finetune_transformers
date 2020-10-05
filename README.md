# finetune_transformers



## 直接调用transformers领域训练model

直接调用transformers来继续领域预训练，下面方法只使用了masked LM 任务，所以只计算了masked LM 任务的损失，来在领域数据上继续优化模型，但是没有使用NSP(或者其他替代方案比如：Sentence Order Prediction)，所以[cls]可能获取不到足够的语义信息，对较多依赖[cls]的任务可能不够友好。

### 准备数据

如果只考虑masked LM 任务，需要准备的数据输入格式可以为一行一篇文档(建议文档长度不超过512)，如果文档长度超过512，可以把文档分成多行，每行可以是多条句子（每行长度不超过512）

### 代码

代码中参数需要根据需求进行配置

```python
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, \
    LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments,RobertaConfig
from sklearn.model_selection import train_test_split
import os
# 配置

transformers_model_name = "hfl/chinese-roberta-wwm-ext"
corpus_file_path = "./data/corpus/all_corpus.txt"
force_download=False
cache_dir='../data/download_transformer_models'
max_sentence_len = 510
random_state=32
valid_size=0.25

config = AutoConfig.from_pretrained(transformers_model_name, type_vocab_size=2,force_download=force_download,cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(transformers_model_name,force_download=force_download,cache_dir=cache_dir )
model = AutoModelForMaskedLM.from_pretrained(transformers_model_name, config=config,force_download=force_download,cache_dir=cache_dir)
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=corpus_file_path, block_size=max_sentence_len )
train_set, valid_set = train_test_split(dataset, test_size=valid_size, random_state=random_state)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
        logging_steps=500,
        overwrite_output_dir=True,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=32,
        max_grad_norm=5.0,
        save_steps=1000,
        save_total_limit=1,
        gradient_accumulation_steps=32,
        evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        eval_steps=1000,

    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=valid_set,

)
trainer.train()

trainer.save("xx")
```

## 自定义 masked 任务及其他任务

有时候，预训练模型所使用的预训练任务在一定程度上难以满足我们的需求，这时就需要根据自己的需求在预训练模型上来自定义预训练任务，然后在领域数据上继续预训练以使我们的模型更加适合下游任务。
本项目以中药说明书命名实体识别数据(如果数据集侵犯版权，请联系删除jeffery.lee.0628@gmail.com)为例，自定义mask lm 任务：对于每句话中的每个实体，50%的概率该实体会被替换成mask(与whole word masking类似，只不过这里mask的是整个实体)。
> 1. 为什么是50%的概率？
     因为：观察数据，句子中的实体数量过多，如果mask全部实体，句子会丢失过多的语义信息，这样就失去了自定义预训练任务的意义，因此，对于每个实体50%的概率会被替换成mask（另外一种方案：被mask住的实体字数不能超过句子长度20%。）。
     
> 2. 由于是ner任务，个人认为token的信息更加重要，所以没有做NSP/SOP 任务，如果下游任务是文本分类，也可以自定义类似NSP/SOP的任务来强化预训练模型。

自定义预训练任务可参考目录data_precess下的python文件中的代码实现。

## 运行
```python
python train.py
```
