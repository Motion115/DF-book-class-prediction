from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# 加载数据
df = pd.read_csv('../data/train.csv')
train_df, val_df = train_test_split(df, test_size=0.1)

# 数据预处理
#将这句话更换为本地文件夹的
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer=RobertaTokenizer.from_pretrained('../model-roberta-base')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

train_encodings = preprocess_function(train_df)
val_encodings = preprocess_function(val_df)

# 创建PyTorch数据集
class BookDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BookDataset(train_encodings, train_df['label'].tolist())
val_dataset = BookDataset(val_encodings, val_df['label'].tolist())

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 初始化模型和训练器
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
#下载到本地的模型
model = RobertaForSequenceClassification.from_pretrained('../model-roberta-base/', num_labels=23)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 训练模型
trainer.train()
#下面是新建的模型预测部分：
# 加载测试数据
test_df = pd.read_csv('test.csv')

# 数据预处理
test_encodings = preprocess_function(test_df)

# 创建PyTorch数据集
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

test_dataset = TestDataset(test_encodings)

# 使用训练好的模型进行预测
'''predictions = trainer.predict(test_dataset=test_dataset)
predicted_labels = torch.argmax(predictions.predictions, dim=-1)

# 将预测结果保存到submission.csv
submission_df = pd.DataFrame({'node_id': test_df['node_id'], 'label': predicted_labels.numpy()})
submission_df.to_csv('submission.csv', index=False)'''


predictions = trainer.predict(test_dataset=test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1)#(torch.tensor(predictions.predictions), dim=-1
submission_df = pd.DataFrame({'node_id': test_df['node_id'], 'label': predicted_labels.numpy()})
submission_df.to_csv('submission1122.csv', index=False)
