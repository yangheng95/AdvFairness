# -*- coding: utf-8 -*-
# file: process_raw.py
# time: 15:06 09/11/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.

import random
from datasets import load_dataset

random.seed(2023)

dataset = load_dataset("fhamborg/news_sentiment_newsmtsc")

f_train = open('train.dat', 'w')
f_test = open('test.dat', 'w')
f_valid = open('valid.dat', 'w')

for i in range(len(dataset['train'])):
    if dataset['train'][i]['polarity'] != 0:
        f_train.write('{} \t{} \t$LABEL${}\n'.format(
            dataset['train'][i]['mention'].replace('\n', ' '),
            dataset['train'][i]['sentence'].replace('\n', ' '),
            "0" if dataset['train'][i]['polarity'] == -1 else "1")
        )

for i in range(len(dataset['test'])):
    if dataset['test'][i]['polarity'] != 0:
        f_test.write('{} \t{} \t$LABEL${}\n'.format(
            dataset['test'][i]['mention'].replace('\n', ' '),
            dataset['test'][i]['sentence'].replace('\n', ' '),
            "0" if dataset['test'][i]['polarity'] == -1 else "1")
        )

for i in range(len(dataset['validation'])):
    if dataset['validation'][i]['polarity'] != 0:
        f_valid.write('{} \t{} \t$LABEL${}\n'.format(
            dataset['validation'][i]['mention'].replace('\n', ' '),
            dataset['validation'][i]['sentence'].replace('\n', ' '),
            "0" if dataset['validation'][i]['polarity'] == -1 else "1")
        )

