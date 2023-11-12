
import json
import zhipuai
import pandas as pd
import re

def clean_json_string(json_str):
    # Remove newlines and extra whitespaces
    cleaned_str = re.sub(r'\s+', ' ', json_str).replace('\n', '').strip()
    # Ensure that keys and values are properly quoted
    cleaned_str = re.sub(r'([{,])\s*([^\"\s]+)\s*:', r'\1 "\2":', cleaned_str)
    cleaned_str = re.sub(r':\s*([^\"\s]+)\s*([,}])', r': "\1"\2', cleaned_str)
    return cleaned_str

prompted_words="Your task is to reorganize the following passage, with the following requirements:\
1. Split the information by Description and Title, and remove the Description and Title prefix.\
2. Make the passage more informative by infering the missing information.\
3. Fix any possible grammatical mistake and spelling mistakes.\
Control the total length of passage in less than 100 words. You should return in a JSON format, with attributes description and title containing the curated content you generated.\
"
modify_list=[]
zhipuai.api_key = "6cd0578b3e951ec8ed7a72092b14ed93.prd14J8G80tVjqlo"
df=pd.read_csv("train_smaller2.csv")
# 初始化两个空列表，用于存储 s1 和 s2 的结果
s1_results = []
s2_results = []
for index,row in df.iterrows():
    texts=row['text']
    texts=prompted_words+texts

    #print(texts)
    response = zhipuai.model_api.sse_invoke(
        model="chatglm_turbo",
        prompt=texts,

    )
    modify_list=[]
    for event in response.events():
            modify_list.append(event.data)
    sentence = ' '.join(modify_list)
    #print(sentence)
    #之前要先预处理，否则json解析失败的
    sentence=clean_json_string(sentence)
    # 提取字符串中的 JSON 部分
    try:
        # 提取字符串中的 JSON 部分
        #json_part = sentence.split('{', 1)[1].rsplit('}', 1)[0]
        #json_content = '{' + json_part + '}'
        # 解析 JSON 内容
        data = json.loads(sentence,strict=False)
        # 将值赋给变量
        s1 = data[' description ']
        s2 = data[' title ']
    except Exception as e:
        print(f"Error processing row {index}: {e}")#
        s1 = s2 = None
        # 将结果添加到列表中
    s1_results.append(s1)
    s2_results.append(s2)
    sentence=""

# 将结果添加到 DataFrame 的新列中
df['s1'] = s1_results
df['s2'] = s2_results
# 去除 'text' 列
df = df.drop(columns=['text'])
# 获取所有列名
columns = df.columns.tolist()
# 将 's1' 和 's2' 列移动到第二列和第三列的位置
columns.insert(1, columns.pop(columns.index('s1')))
columns.insert(2, columns.pop(columns.index('s2')))
# 重新排序列
df = df[columns]
# 将修改后的 DataFrame 写回到 CSV 文件
df.to_csv("train__smaller_modified02.csv", index=False)