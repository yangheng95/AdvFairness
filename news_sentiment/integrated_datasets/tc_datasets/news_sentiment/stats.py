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
from  datasets import load_dataset
import random
random.seed(2023)

data = load_dataset("fhamborg/news_sentiment_newsmtsc")
negative_reviews = []
positive_reviews = []
dataset = data['train']
for i in range(len(dataset)):
    if dataset[i]['polarity'] > 0:
        negative_reviews.append([dataset[i]['mention'], dataset[i]['sentence'], dataset[i]['polarity']])
    else:
        positive_reviews.append([dataset[i]['mention'], dataset[i]['sentence'], dataset[i]['polarity']])

target_negative = {}
for i in range(len(negative_reviews)):
    if negative_reviews[i][0] in target_negative:
        target_negative[negative_reviews[i][0]] += 1
    else:
        target_negative[negative_reviews[i][0]] = 1

target_positive = {}
for i in range(len(positive_reviews)):
    if positive_reviews[i][0] in target_positive:
        target_positive[positive_reviews[i][0]] += 1
    else:
        target_positive[positive_reviews[i][0]] = 1

target_negative = sorted(target_negative.items(), key=lambda item: item[1], reverse=True)
target_positive = sorted(target_positive.items(), key=lambda item: item[1], reverse=True)

# calculate intersection
target_negative_tuple = dict(target_negative)
target_positive_tuple = dict(target_positive)

target_negative_set = set(target_negative_tuple.keys())
target_positive_set = set(target_positive_tuple.keys())

print(target_negative_set & target_positive_set)


with open('negative_targets.txt', 'w', encoding='utf8') as f:
    for item in target_negative_set:
        f.write(str(item) + '\n')

with open('positive_targets.txt', 'w', encoding='utf8') as f:
    for item in target_positive_set:
        f.write(str(item) + '\n')

with open('all_targets.txt', 'w', encoding='utf8') as f:
    for item in target_negative_set | target_positive_set:
        f.write(str(item) + '\n')

print(len(target_negative_set & target_positive_set))

print(target_negative[:100])
print(target_positive[:100])

#
target_negative_tuple = dict(target_negative[:10])
target_positive_tuple = dict(target_positive[:10])
import matplotlib.pyplot as plt

plt.bar([i for i in range(len(target_negative_tuple.keys()))], target_negative_tuple.values())
plt.xticks([i for i in range(len(target_negative_tuple.keys()))], target_negative_tuple.keys(), rotation=90)
plt.xlabel('target')
plt.ylabel('frequency')
plt.title('Histogram of target in negative_news')
plt.tight_layout()
plt.show()

plt.bar([i for i in range(len(target_positive_tuple.keys()))], target_positive_tuple.values())
plt.xticks([i for i in range(len(target_positive_tuple.keys()))], target_positive_tuple.keys(), rotation=90)
plt.xlabel('target')
plt.ylabel('frequency')
plt.title('Histogram of target in positive_news')
plt.tight_layout()
plt.show()
#
#
# print(len(negative_news))
# print(len(positive_news))
# print(negative_news[0])
# # print(data.iloc[i]['id'])
# # print(data.iloc[i]['author'])
# # print(data.iloc[i]['title'])
# # print(data.iloc[i]['text'])
# # print(data.iloc[i]['label'])
# # print()
