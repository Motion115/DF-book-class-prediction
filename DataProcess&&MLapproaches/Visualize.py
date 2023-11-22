import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from wordcloud import WordCloud

# 加载数据
df = pd.read_csv('../data/train.csv')
#print(df.head())
# 直方图
plt.figure(figsize=(10, 6))
sns.histplot(df['label'])
plt.title('Label Distribution')
#plt.show()
plt.savefig('filename1.png')
# 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='node_id', y='label', data=df)
plt.title('Scatter Plot of Node ID vs Label')
#plt.show()
plt.savefig('filename2.png')
# 箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='node_id', data=df)
plt.title('Boxplot of Labels')
#plt.show()
plt.savefig('filename3.png')

# 计数图
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Count of Each Label')
#plt.show()
plt.savefig('filename4.png')
# 条形图
label_counts = df['label'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Bar Chart of Label Counts')
#plt.show()
plt.savefig('filename5.png')
# 饼图
plt.figure(figsize=(8, 8))
df['label'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Pie Chart of Labels')
#plt.show()
plt.savefig('filename6.png')
# 词云
text = " ".join(description for description in df.text)
wordcloud = WordCloud(background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Descriptions')
#plt.show()
plt.savefig('filename7.png')
