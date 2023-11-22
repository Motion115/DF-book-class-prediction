import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification
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
model_path = "results/checkpoint-5000/"  # 模型所在的文件夹
model = RobertaForSequenceClassification.from_pretrained(model_path)
import pandas as pd

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('../model-roberta-base/')

# 对数据进行tokenize处理 ,max_length=512, truncation=True, padding=True)


# 定义了 preprocess_function 和 TestDataset
test_df = pd.read_csv('../data/test.csv')
#这里可以根据词云的分析结果加一些后处理，
#test_encodings = preprocess_function(test_df)
test_encodings = tokenizer(test_df['text'].tolist(), max_length=512,truncation=True, padding=True)
#test_dataset = TestDataset(test_encodings)
test_dataset = BookDataset(test_encodings)
# 使用训练好的模型进行预测
from transformers import Trainer
trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset=test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1)#(torch.tensor(predictions.predictions), dim=-1

# 将预测结果保存到submission.csv
'''submission_df = pd.DataFrame({'node_id': test_df['node_id'], 'label': predicted_labels.numpy()})
submission_df.to_csv('submission1122.csv', index=False)'''
# 后处理：如果特定词汇出现次数超过阈值，则更改预测结果
# 定义标签与特殊关键词的映射
label_keywords = {
    0: ["literature"],
    1: ["cat", "dog"],
    2: ["grow"],
    3: ["humor", "funny", "joke"],
    4: ["train", "car", "trunk"],
    5: ["fairy", "myth"],
    6: ["activity", "activities", "craft", "game"],
    7: ["science", "fantasy"],
    8: ["classic"],
    9: ["mystery", "detective"],
    10: ["action"],
    11: ["geography", "culture"],
    12: ["education", "reference"],
    13: ["art", "music", "photography"],
    14: ["holiday", "celebration"],
    15: ["science", "nature"],
    16: ["learning"],
    17: ["biographies"],
    18: ["history"],
    19: ["recipe", "cookbook", "cook", "cooking"],
    20: ["bible", "god", "religion"],
    21: ["sports", "baseball", "sport", "outdoor", "team", "ball"],
    22: ["comics", "graphic"],
    23: ["computer", "technology", "coding"]
}

# 后处理逻辑
threshold = 2
label_changes = {label: 0 for label in label_keywords.keys()}

for i, text in enumerate(test_df['text']):
    text_lower = text.lower()  # 转换为小写
    for label, keywords in label_keywords.items():
        if any(text_lower.count(keyword) >= threshold for keyword in keywords):
            if predicted_labels[i] != label:
                label_changes[label] += 1
            predicted_labels[i] = label
            break

# 输出每个标签修改的数量
for label, count in label_changes.items():
    print(f"Label {label} changes: {count}")

# 保存修改后的预测结果
submission_df = pd.DataFrame({'node_id': test_df['node_id'], 'label': predicted_labels})
submission_df.to_csv('submission1122_new.csv', index=False)

