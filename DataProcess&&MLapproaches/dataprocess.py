import pandas as pd
import re
import string

def clean_text(text):
    # 转换为小写
    text = text.lower()
    
    # 移除网址
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 移除邮箱地址
    text = re.sub(r'\S*@\S*\s?', '', text)
    #去除des和title等字眼：
    text = re.sub(r"description:", "", text)
    text = re.sub(r"title:", "",text)
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 移除数字
    text = re.sub(r'\d+', '', text)

    # 移除特殊字符
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # 移除日期（这里需要根据实际格式进行调整）
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)  # 示例：移除 YYYY-MM-DD 格式的日期

    # 移除数字（如果需要）
    text = re.sub(r'\d+', '', text)

    # 移除额外的空格、换行符等
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    return text

# 加载数据
df = pd.read_csv('../data/test.csv')
#df = pd.read_csv('../data/test.csv')
#df['Description'] = ''
#df['Title'] = ''
# 遍历 DataFrame，分割 Description 和 Title
'''for index, row in df.iterrows():
    text = row['text']
    # 使用正则表达式分割
    #split_text = re.split(r'Title:', text)
    df.at[index, 'Description'] = split_text[0].replace('Description:', '').strip()
    if len(split_text) > 1:
        df.at[index, 'Title'] = split_text[1].strip()
        df.at[index, 'Title'] = split_text[1].replace('Title:', '').strip()'''
# 清理文本列
df['text'] = df['text'].apply(clean_text)
#df['Title']=df['Title'].apply(clean_text)

#df.drop('cleaned_text',axis=1,inplace=True)
# 可以选择保存清理后的数据
df.to_csv('test_cleaned.csv', index=False)
