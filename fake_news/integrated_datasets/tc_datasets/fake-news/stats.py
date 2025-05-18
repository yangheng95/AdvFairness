# -*- coding: utf-8 -*-
# file: process_raw.py
# time: 14:10 03/11/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.

import pandas
import findfile

fake_news = []
true_news = []
f_name = 'train.csv.ffi'
data = pandas.read_csv(f_name)
data.dropna()
for i in range(len(data)):
    if data.iloc[i]['title'] == "nan" or data.iloc[i]['text'] == "nan" or data.iloc[i]['author'] == "nan":
        continue
    if data.iloc[i]['label'] == 1:
        fake_news.append([data.iloc[i]['id'], data.iloc[i]['title'], data.iloc[i]['text'], data.iloc[i]['author'],
                          data.iloc[i]['label']])
    else:
        true_news.append([data.iloc[i]['id'], data.iloc[i]['title'], data.iloc[i]['text'], data.iloc[i]['author'],
                          data.iloc[i]['label']])


authors_fake = {}
for i in range(len(fake_news)):
    if fake_news[i][-2] in authors_fake:
        authors_fake[fake_news[i][-2]] += 1
    else:
        authors_fake[fake_news[i][-2]] = 1

authors_true = {}
for i in range(len(true_news)):
    if true_news[i][-2] in authors_true:
        authors_true[true_news[i][-2]] += 1
    else:
        authors_true[true_news[i][-2]] = 1

authors_fake = sorted(authors_fake.items(), key=lambda item: item[1], reverse=True)
authors_true = sorted(authors_true.items(), key=lambda item: item[1], reverse=True)

# calculate intersection
authors_fake_tuple = dict(authors_fake)
authors_true_tuple = dict(authors_true)

authors_fake_set = set(authors_fake_tuple.keys())
authors_true_set = set(authors_true_tuple.keys())

print(authors_fake_set & authors_true_set)


with open('negative_targets.txt', 'w', encoding='utf8') as f:
    for item in authors_fake_set:
        f.write(str(item) + '\n')

with open('positive_targets.txt', 'w', encoding='utf8') as f:
    for item in authors_true_set:
        f.write(str(item) + '\n')

with open('all_targets.txt', 'w', encoding='utf8') as f:
    for item in authors_fake_set | authors_true_set:
        f.write(str(item) + '\n')

print(len(authors_fake_set & authors_true_set))

print(authors_fake[:100])
print(authors_true[:100])

#
authors_fake_tuple = dict(authors_fake[1:11])
authors_true_tuple = dict(authors_true[:10])
import matplotlib.pyplot as plt

plt.bar([i for i in range(len(authors_fake_tuple.keys()))], authors_fake_tuple.values())
plt.xticks([i for i in range(len(authors_fake_tuple.keys()))], authors_fake_tuple.keys(), rotation=90)
plt.xlabel('authors')
plt.ylabel('frequency')
plt.title('Histogram of authors in fake_news')
plt.tight_layout()
plt.show()

plt.bar([i for i in range(len(authors_true_tuple.keys()))], authors_true_tuple.values())
plt.xticks([i for i in range(len(authors_true_tuple.keys()))], authors_true_tuple.keys(), rotation=90)
plt.xlabel('authors')
plt.ylabel('frequency')
plt.title('Histogram of authors in true_news')
plt.tight_layout()
plt.show()
#
#
# print(len(fake_news))
# print(len(true_news))
# print(fake_news[0])
# # print(data.iloc[i]['id'])
# # print(data.iloc[i]['author'])
# # print(data.iloc[i]['title'])
# # print(data.iloc[i]['text'])
# # print(data.iloc[i]['label'])
# # print()
