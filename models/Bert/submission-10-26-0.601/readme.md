## 模型
1. 仅使用文本特征
2. BertForSequenceClassification

## GPU
1. V100-32G

## 改进
1. 添加 '[CLS]' 和 '[SEP]'
2. max_len = 512
3. 数据集手动删除无用的token

## 复现方式
1. best_model.pth为在验证集上的最佳模型，run predict.py即可在测试集上预测label
