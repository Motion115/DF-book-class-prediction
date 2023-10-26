import pandas as pd 
import numpy as np 
import json, time 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
import csv
warnings.filterwarnings('ignore')


bert_path = "bert_model/"    # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = BertTokenizer.from_pretrained(bert_path)   # 初始化分词器

print("done")

maxlen = 320  

#读取测试集
input_ids_test, input_masks_test, input_types_test,  = [], [], []
with open("test.csv", encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        if len(row) >= 1:  # 确保每行至少有两个单元格
            title1 = row[0]
            

        # encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
        # 根据参数会短则补齐，长则切断
        encode_dict_test = tokenizer.encode_plus(text=title1, max_length=maxlen, 
                                            padding='max_length', truncation=True)
        
        input_ids_test.append(encode_dict_test['input_ids'])
        input_types_test.append(encode_dict_test['token_type_ids'])
        input_masks_test.append(encode_dict_test['attention_mask'])

        #labels.append(int(y))

input_ids_test, input_types_test, input_masks_test = np.array(input_ids_test), np.array(input_types_test), np.array(input_masks_test)
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape)


print("test_load_done")


BATCH_SIZE=16
# 测试集（是没有标签的）
test_data = TensorDataset(torch.LongTensor(input_ids_test), 
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(input_types_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)



# 定义model
class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=24):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)     # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]   # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)   #  [bs, classes]
        return logit


def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Bert_Model(bert_path).to(DEVICE)
print(get_parameter_number(model))



# 测试集没有标签
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred


# 加载最优权重对测试集测试
model.load_state_dict(torch.load("best_bert_model.pth"))
pred_test = predict(model, test_loader, DEVICE)

# 指定 CSV 文件名
csv_file = 'predictions.csv'

# 打开 CSV 文件以写入
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    for prediction in pred_test:
        writer.writerow([prediction])
#print("\n Test Accuracy = {} \n".format(accuracy_score(y_test, pred_test)))
#print(classification_report(y_test, pred_test, digits=4))
