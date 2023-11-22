import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')

# 加载数据
#df = pd.read_csv('../data/train.csv')
df = pd.read_csv('train_cleaned.csv')
# 1. 词频分布图
from collections import Counter
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
df = df.dropna(subset=['Title', 'Description'])
# 统计词频
words = word_tokenize(" ".join(df['Description'].tolist()))
word_freq = Counter(words)

# 取最常见的10个词
common_words = word_freq.most_common(10)
words_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='word', data=words_df)
plt.title('Top 10 Common Words')
plt.savefig('word_frequency.png')
plt.clf()

# 2. 类别标签分布图
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Label Distribution')
plt.savefig('label_distribution.png')
plt.clf()

# 3. 文本长度分布图
df['text_length'] = df['Description'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=30)
plt.title('Description Length Distribution')
plt.savefig('Description_length_distribution.png')
plt.clf()

# 4. 类别与文本长度关系图
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='text_length', data=df)
plt.title('Description Length per Label')
plt.savefig('Description_length_per_label.png')
plt.clf()

# 5. 类别与节点编号关系图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='node_id', y='label', data=df)
plt.title('Label vs Node ID')
plt.savefig('label_vs_node_id.png')
plt.clf()

label_dic={0:"Literature & Fiction",1:"Animals",2:"Growing Up & Facts of Life",  3:"Humor",4:"Cars, Trains & Things That Go",
           5:"Fairy Tales, Folk Tales & Myths",6:"Activities, Crafts & Games",7:"Science Fiction & Fantasy",8:"Classics",9:"Mysteries & Detectives",
           10:"Action & Adventure",11:"Geography & Cultures",12:"Education & Reference",13:"Arts, Music & Photography",14:"Holidays & Celebrations",
           15:"Science, Nature & How It Works",16:"Early Learning",17:"Biographies",18:"History",19:"Children's Cookbooks",
           20:"Religions",21:"Sports & Outdoors",22:"Comics & Graphic Novels",23:"Computers & Technology"
}

# 6. 词云图（按类别）
for label in df['label'].unique():
    subset = df[df['label'] == label]
    text = " ".join(description for description in subset.Description)
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Label {label}:{label_dic[label]}')#modify the name of the png
    plt.savefig(f'wordcloud_label_{label}_{label_dic[label]}.png')
    plt.clf()#用来清除已经有的图
