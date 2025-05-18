# -*- coding: utf-8 -*-
# file: process_raw.py
# time: 12:25 06/11/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import random

import findfile
import pandas

data_items = []

f = 'train.csv.ffi'
data_frame = pandas.read_csv(f)
print(data_frame.columns)
data_frame.dropna(inplace=True)

for i in range(len(data_frame)):
    if data_frame.iloc[i]['title'] == "nan" or data_frame.iloc[i]['text'] == "nan" or data_frame.iloc[i]['author'] == "nan":
        continue
    data_items.append([data_frame.iloc[i]['id'], data_frame.iloc[i]['author'], data_frame.iloc[i]['title'],
                       data_frame.iloc[i]['text'], data_frame.iloc[i]['label']])

# split dataset into train, valid and test set
random.seed(2023)
random.shuffle(data_items)

with open('train.dat', 'w', encoding='utf8') as f:
    for item in data_items[int(0.2 * len(data_items)):]:
        f.write('{} \t{} \t{}$LABEL${}'.format(item[1], item[2], str(item[3])[:500], item[4]).replace(
            '\n', '') + '\n')

with open('test.dat', 'w', encoding='utf8') as f:
    for item in data_items[int(0.1 * len(data_items)):int(0.2 * len(data_items))]:
        f.write('{} \t{} \t{}$LABEL${}'.format(item[1], item[2], str(item[3])[:500], item[4]).replace(
            '\n', '') + '\n')

with open('valid.dat', 'w', encoding='utf8') as f:
    for item in data_items[:int(0.1 * len(data_items))]:
        f.write('{} \t{} \t{}$LABEL${}'.format(item[1], item[2], str(item[3])[:500], item[4]).replace(
            '\n', '') + '\n')
