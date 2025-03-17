import pandas as pd
import numpy as np
import re
from collections import defaultdict

# 读取情感分类数据集（假设文件路径和字段）
data = pd.read_csv("data/sent/sentiment_data.csv")

print("前几条数据：")
print(data.head(5))

# 假设文本列名为'text'
sentences = np.array(data['text'])
print(f"\n句子总数: {len(sentences)}")

# 清洗文本：去掉用户名、网址等无关内容
pattern = r'@\w+[:]?|http\S+'
clean_sentences = []
for sentence in sentences:
    clean = re.sub(pattern, '', sentence).strip()
    clean = re.sub(' +', ' ', clean)
    clean_sentences.append(clean)

# 更新清洗后的文本
data['text'] = clean_sentences

# 定义受保护属性正则表达式
def contains_protected(sentence):
    protected_factors = {
        "race_ethnicity": [
            r"\b(black|white|asian|hispanic|latino|african american|native american|caucasian|arab|indian|pacific islander|middle eastern)\b"
        ],
        "gender": [
            r"\b(man|woman|male|female|boy|girl|he|she|him|her|transgender|trans|non-binary|genderqueer)\b"
        ],
        "sexual_orientation": [
            r"\b(gay|lesbian|bisexual|homosexual|heterosexual|queer|pansexual|asexual)\b"
        ],
        "religion": [
            r"\b(christian|muslim|jewish|jew|buddhist|hindu|sikh|atheist|agnostic|catholic|protestant)\b"
        ],
        "disability": [
            r"\b(disabled|disability|handicapped|impairment|autism|autistic|blind|deaf|wheelchair)\b"
        ],
        "age": [
            r"\b(child|children|teen|teenager|young adult|adult|elderly|senior citizen|old man|old woman)\b",
            r"\b(\d{1,3}\s*(years old|yo))\b"   # 检测年龄描述
        ],
        "nationality": [
            r"\b(american|british|chinese|japanese|mexican|french|german|canadian|indian|russian|korean)\b"
        ]
    }

    found_protected = {}
    for category, patterns in protected_factors.items():
        found_protected[category] = []
        for pat in patterns:
            if re.search(pat, sentence, re.IGNORECASE):
                found_protected[category].append(pat)
    return found_protected

# 遍历检测
protected_counts = defaultdict(int)
protected_row_numbers = []
protected_categories = []

for index, sentence in enumerate(clean_sentences):
    protected = contains_protected(sentence)
    if any(protected.values()):
        protected_row_numbers.append(index)
        for category, patterns in protected.items():
            if patterns:
                protected_categories.append(category)
                protected_counts[category] += 1
                break  # 只记录第一个出现的类别

# 输出统计信息
print("\n受保护属性关键词统计：")
for category, count in protected_counts.items():
    print(f"{category}: {count}")

# 提取包含受保护属性的句子
protected_data = data.iloc[protected_row_numbers]
protected_data['protected_category'] = np.array(protected_categories)

print(f"\n检测到受保护属性的样本数量: {protected_data.shape[0]}")
print(protected_data.head(10))



# 可以选择保存结果
# protected_data.to_csv("data/sentiment/protected_data.csv", index=False)

# # （可选）对类别做进一步分析
# def adjust_class_labels(row):
#     # 假设原始数据里有情感标签 'label'，值为 0（负面）、1（中性）、2（正面）
#     if row['label'] in [0, 1]:
#         return 1  # 负面或中性都记为 1
#     elif row['label'] == 2:
#         return 0  # 正面记为 0
#     else:
#         raise ValueError("标签值异常")
#
# protected_data['adjusted_label'] = protected_data.apply(adjust_class_labels, axis=1)
#
# # 分类别统计正负样本数量
# print("\n受保护属性类别和情感标签分布：")
# print(protected_data.groupby(['protected_category', 'adjusted_label']).size())
#
# # 平衡数据集（可选）
# def balance_dataframe(df, group_cols, label_col):
#     balanced_df_list = []
#     for category, group in df.groupby(group_cols):
#         class_counts = group[label_col].value_counts()
#         min_count = class_counts.min()
#
#         balanced_group = group.groupby(label_col).apply(lambda x: x.sample(min_count, random_state=42))
#         balanced_group = balanced_group.reset_index(drop=True)
#         balanced_df_list.append(balanced_group)
#
#     balanced_df = pd.concat(balanced_df_list, ignore_index=True)
#     return balanced_df
#
# balanced_df = balance_dataframe(protected_data, group_cols='protected_category', label_col='adjusted_label')
#
# print("\n平衡后的数据分布：")
# print(balanced_df.groupby(['protected_category', 'adjusted_label']).size().reset_index(name='count'))
#
# # 保存平衡后的数据
# # balanced_df.to_csv("data/sentiment/balanced_protected_data.csv", index=False)
#
# print(f"\n平衡后的样本数量: {balanced_df.shape[0]}")
