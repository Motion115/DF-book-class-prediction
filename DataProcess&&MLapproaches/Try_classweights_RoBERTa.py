import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np

# 加载数据并做预处理
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.1
)

# 重置索引
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# 初始化tokenizer
tokenizer = RobertaTokenizer.from_pretrained('../model-roberta-base/')

# 对数据进行tokenize处理
train_encodings = tokenizer(train_texts.tolist(), max_length=512, truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), max_length=512, truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), max_length=512, truncation=True, padding=True)

# 定义数据集
class BookDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = BookDataset(train_encodings, train_labels)
val_dataset = BookDataset(val_encodings, val_labels)
test_dataset = BookDataset(test_encodings)

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 初始化模型
model = RobertaForSequenceClassification.from_pretrained(
    '../model-roberta-base/',
    num_labels=len(train_df['label'].unique())
)

# 设置模型的分类器权重 error del
#model.classifier.out_proj.weight.data = model.classifier.out_proj.weight.data.new(class_weights.view(-1, 1))

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=6,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)
# 确定模型所在的设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 将类别权重移到相同的设备
class_weights = class_weights.to(device)
from torch.nn import CrossEntropyLoss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 现在使用 CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
# 定义带有类别权重的损失函数
'''def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss
'''
# 创建训练器时使用自定义的损失函数

'''trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,     
    #compute_loss=compute_loss,  # 使用自定义的损失函数        
)'''

# 开始训练
trainer.train()

# 预测
predictions = trainer.predict(test_dataset)
#pred_labels = predictions.predictions.argmax(-1)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1)#(torch.tensor(predictions.predictions), dim=-1
# 将预测结果保存到submission.csv
submission_df = pd.DataFrame({'node_id': test_df['node_id'], 'label': predicted_labels.numpy()})
submission_df.to_csv('submission03.csv', index=False)
