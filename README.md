# finetune_transformers



## 直接调用transformers领域训练model

直接调用transformers来继续领域预训练，下面方法只使用了masked LM 任务，所以只计算了masked LM 任务的损失，来在领域数据上继续优化模型，但是没有使用NSP(或者其他替代方案比如：Sentence Order Prediction)，所以[cls]可能获取不到足够的语义信息，对较多依赖[cls]的任务可能不够友好。

### 准备数据

如果只考虑masked LM 任务，需要准备的数据输入格式可以为一行一篇文档(建议文档长度不超过512)，如果文档长度超过512，可以把文档分成多行，每行可以是多条句子（建议每行长度不超过512）

### 代码

代码中参数需要根据需求进行配置

```python
from transformers import AutoConfig, AutoModelForMaskedLM,AutoTokenizer,LineByLineTextDataset,DataCollatorForLanguageModeling,Trainer, TrainingArguments


tokenizer = AutoTokenizer.from_pretrained(transformers_model_name,max_len=512)
config = AutoConfig.from_pretrained(transformers_model_name,max_position_embeddings=514,type_vocab_size=1)
model = AutoModelForMaskedLM.from_pretrained(transformers_model_name,config=config)
dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path=corpus_file_path,block_size=512,)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,

)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,

)

trainer.train()

if __name__ == '__main__':
    transformers_model_name = "hfl/chinese-roberta-wwm-ext"
    corpus_file_path = "./corpus.txt"
    finetune_model(transformers_model_name,corpus_file_path)
```



## 自定义 masked 任务及其他任务

有时候，预训练模型所使用的预训练任务在一定程度上难以满足我们的需求，这时我们就需要根据自己的需求在预训练模型上来自定义预训练任务，然后在领域数据上继续预训练以使我们的模型更加适合下游任务。





