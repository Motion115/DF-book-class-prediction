#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!pip install nltk

import nltk
#nltk.download('stopwords', download_dir='../') #如果没下载过需要去掉注释


# In[5]:


get_ipython().system('unzip corpora/wordnet.zip -d corpora/')


# In[8]:


#nltk.download('wordnet',download_dir='../')


# In[20]:


# 自定义函数
def my_function(x):
    return x * 2

# 创建一个DataFrame对象
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# 使用.apply()函数应用自定义函数
result = df.apply(my_function)
print(result)


# In[11]:


import nltk
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

# 分别定义需要进行还原的单词与相对应的词性
words = ['cars','men','running','ate','saddest','fancier']
pos_tags = ['n','n','v','v','a','a']

for i in range(len(words)):
    print(words[i]+'--'+pos_tags[i]+'-->'+wnl.lemmatize(words[i],pos_tags[i]))


# In[ ]:





# In[14]:


from nltk.corpus import stopwords 
stop = set(stopwords.words('english')) 
print(stop)


# In[15]:


import re
def preprocess_text(text):
    #去除句子首部的describtion：
    text = text.lstrip("Description:")
    # 小写转换
    text = text.lower()
    
    # 去除标点符号和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 分词
    words = text.split()
    
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 词干提取或词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 过滤器：根据特定的条件过滤掉一些词语，例如长度小于2的词语或者只包含数字的词语等。
    #words = [word for word in words if len(word) > 1 or not word.isdigit()]
    
    return ' '.join(words)
test="hi,2343apple I like best"
rert=preprocess_text(test)
print(rert)


# In[16]:


#去除停用词
sentence = "this is a apple"
filter_sentence= [w for w in sentence.split(' ') if w not in stopwords.words('english')]
print(filter_sentence)


# In[28]:


import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from nltk.stem import WordNetLemmatizer
# 1. 数据加载和预处理
data = pd.read_csv('../train.csv')
X = data['text']  # 文本数据
print(type(X))
y = data['label']  # 类别标签
# 2. 文本预处理 输入的参数text是每条description的记录
def preprocess_text(text):
    #去除句子首部的describtion：
    text = text.lstrip("Description:")
    # 小写转换
    text = text.lower()
    
    # 去除标点符号和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 分词
    words = text.split()
    
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 词干提取或词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 过滤器：根据特定的条件过滤掉一些词语，例如长度小于2的词语或者只包含数字的词语等。
    words = [word for word in words if len(word) > 1 or not word.isdigit()]
    
    return ' '.join(words)





# In[29]:


#对train.csv进行文本处理，去除停用词、词形还原
# 应用文本预处理函数
data['text'] = data['text'].apply(preprocess_text)#.apply函数用于对每一行都执行preprocess_text函数

# 将处理后的结果写回到源文件
data.to_csv('train_processed_all.csv', index=False)


# In[30]:


#数据读入 2023/10/29

# 1. 数据加载和预处理
data = pd.read_csv('train_processed_all.csv')
X = data['text']  # 文本数据
print(type(X))
y = data['label']  # 类别标签


# In[31]:


#apply函数是对df中的每个元素
X = X.apply(preprocess_text)
# 2. 文本向量化
vectorizer = TfidfVectorizer(max_features=1000)  # 使用TF-IDF向量化文本
X = vectorizer.fit_transform(X)

# 3. 数据划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 模型选择和训练
# 使用多种分类方法
models = {
    'MultinomialNB': MultinomialNB(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree':DecisionTreeClassifier(),
    'RandomForest':RandomForestClassifier(),
    'GradientBoosting':GradientBoostingClassifier()
}

for model_name, model in models.items():
    model.fit(X, y)
    # 5. 模型评估
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'{model_name} Validation Accuracy: {accuracy}')

# 6. 模型预测
test_data = pd.read_csv('../test.csv')
X_test = vectorizer.transform(test_data['text'])

# 针对每个模型进行预测
predictions = {}
for model_name, model in models.items():
    y_test_pred = model.predict(X_test)
    predictions[model_name] = y_test_pred

# 7. 结果保存
for model_name, y_test_pred in predictions.items():
    submission = pd.DataFrame({'node_id': test_data['node_id'], 'label': y_test_pred})
    submission.to_csv(f'submission1029_{model_name}.csv', index=False)
    


# In[ ]:




