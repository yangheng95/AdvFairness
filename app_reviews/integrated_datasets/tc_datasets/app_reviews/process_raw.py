# -*- coding: utf-8 -*-
# file: prepare_raw.py
# time: 14:35 09/11/2023
# author: YANG, HENG
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import random
from datasets import load_dataset

random.seed(2023)
dataset = load_dataset("app_reviews")

f_train = open('train.dat', 'w')
f_test = open('test.dat', 'w')
f_valid = open('valid.dat', 'w')

# write train set
for i in range(len(dataset['train'])):
    if random.random() < 0.8:
        f_train.write('{} \t {} \t$LABEL${}\n'.format(
            dataset['train'][i]['package_name'].replace('\n', ' '),
            dataset['train'][i]['review'].replace('\n', ' '),
            "0" if dataset['train'][i]['star'] < 4 else "1")
        )
    elif random.random() < 0.9:
        f_valid.write('{} \t {} \t$LABEL${}\n'.format(
            dataset['train'][i]['package_name'].replace('\n', ' '),
            dataset['train'][i]['review'].replace('\n', ' '),
            "0" if dataset['train'][i]['star'] < 4 else "1")
        )
    else:
        f_test.write('{} \t {} \t$LABEL${}\n'.format(
            dataset['train'][i]['package_name'].replace('\n', ' '),
            dataset['train'][i]['review'].replace('\n', ' '),
            "0" if dataset['train'][i]['star'] < 4 else "1")
        )
