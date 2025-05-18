# -*- coding: utf-8 -*-
# file: pred_utils.py
# time: 16:55 06/11/2023
# author: YANG, HENG <hy345@exeter.ac.uk> 
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import copy
import os.path

import findfile
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing


def load_encoded_protected_attributes(protected_attributes, dataset_name):
    """加载并编码受保护的属性"""
    positive_targets = load_positive_targets(dataset_name)
    # negative_targets = load_negative_targets(dataset_name)
    all_targets = load_all_targets(dataset_name)
    # 为每个受保护属性创建一个唯一的编码
    protected_attributes = [attr.strip() for attr in protected_attributes]
    unique_protected_attributes = set(protected_attributes)
    encoder = {author: i for i, author in enumerate(unique_protected_attributes)}

    # 分别为有利和不利群体创建编码字典
    privileged_groups = [{'protected_attribute': encoder[author]} for author in positive_targets if author in encoder]
    unprivileged_groups = [{'protected_attribute': encoder[author]} for author in all_targets if author in encoder if author not in positive_targets]

    return privileged_groups, unprivileged_groups, encoder

def create_binary_label_dataset(labels, protected_attributes, encoder):
    """根据标签和受保护属性创建BinaryLabelDataset对象"""

    # 将字符串标签转换为二进制
    binary_labels = [int(l) for l in labels]
    protected_attributes = [attr.strip() for attr in protected_attributes]

    # 创建DataFrame
    df = pd.DataFrame({
        'label': binary_labels,
        'protected_attribute': [encoder[author] for author in protected_attributes]
    })

    # 创建BinaryLabelDataset对象
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=['label'],
        protected_attribute_names=['protected_attribute']
    )

def calculate_fairness_metrics(true_dataset, pred_dataset, unprivileged_groups, privileged_groups):
    """计算并返回公平性度量"""
    metric = ClassificationMetric(true_dataset, pred_dataset,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    return {
        'mean_difference': metric.mean_difference(),
        'disparate_impact': metric.disparate_impact(),
        'statistical_parity_difference': metric.statistical_parity_difference()
    }

def calculate_overall_disparity(true_labels, predicted_labels, protected_attributes, dataset_name):
    # Load encoded protected attributes and their groups
    privileged_groups, unprivileged_groups, encoder = load_encoded_protected_attributes(protected_attributes, dataset_name)

    # Create datasets
    true_dataset = create_binary_label_dataset(true_labels, protected_attributes, encoder)
    pred_dataset = create_binary_label_dataset(predicted_labels, protected_attributes, encoder)

    # Ensure that datasets differ only in labels
    # The protected attributes, instance weights, and features should be identical.
    pred_dataset.features = true_dataset.features
    pred_dataset.protected_attributes = true_dataset.protected_attributes
    pred_dataset.instance_weights = true_dataset.instance_weights
    pred_dataset.unfavorable_label = true_dataset.unfavorable_label
    pred_dataset.favorable_label = true_dataset.favorable_label

    # Apply reweighing to the predicted dataset
    reweighing = Reweighing(unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
    pred_dataset = reweighing.fit_transform(pred_dataset)

    # Compute fairness metrics
    fairness_metrics = calculate_fairness_metrics(true_dataset, pred_dataset,
                                                  unprivileged_groups, privileged_groups)

    return fairness_metrics

def calculate_binary_dataset_disparity(true_labels, protected_attributes, dataset_name):
    privileged_groups, unprivileged_groups, encoder = load_encoded_protected_attributes(protected_attributes, dataset_name)
    dataset = create_binary_label_dataset(true_labels, protected_attributes, encoder)

    # 计算度量指标
    metric = BinaryLabelDatasetMetric(dataset,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
    mean_difference = metric.mean_difference()
    disparate_impact = metric.disparate_impact()
    statistical_parity_difference = metric.statistical_parity_difference()

    results = {
        'mean_difference': mean_difference,
        'disparate_impact': disparate_impact,
        'statistical_parity_difference': statistical_parity_difference
    }

    return results

def predict_dataset(victim_classifier, pred_news_tuple=None):
    preds = []
    targets = []
    for (target, text) in pred_news_tuple:
        targets.append(target)
        result = victim_classifier.predict(text, print_result=False)
        preds.append(result['label'])
    return preds, targets


def evaluate_performance(victim_classifier, pred_news_tuple=None):
    news = [item[1] for item in pred_news_tuple]
    results = victim_classifier.predict(news, print_result=False)
    return results


def prepare_test_dataset(dataset_name='fake-news'):
    test_set_file = findfile.find_cwd_files(['test', '.dat'])
    test_set_tuple = []

    for f in test_set_file:
        with open(f, 'r', encoding='utf8') as test_set:
            for line in test_set.readlines():
                line = line.strip()
                if line:
                    test_set_tuple.append((line.split('\t')[0].replace('Target:', ''), line))

    return test_set_tuple


def load_positive_targets(dataset_name):
    positive_targets = []
    with open(findfile.find_cwd_file(['positive_targets.txt', dataset_name]), 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                positive_targets.append(line)
    return positive_targets


def load_negative_targets(dataset_name):
    negative_targets = []
    with open(findfile.find_cwd_file(['negative_targets.txt', dataset_name]), 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                negative_targets.append(line)
    return negative_targets


def load_all_targets(dataset_name):
    all_targets = []
    with open(findfile.find_cwd_file(['all_targets.txt', dataset_name]), 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                all_targets.append(line)
    return all_targets


if __name__ == '__main__':
    test_set_tuple = prepare_test_dataset(dataset_name='fake-news')

    # with open('targets/positive_targets.txt', 'w', encoding='utf8') as f_positive:
    #     for author, line in test_set_tuple.items():
    #         if line.split('$LABEL$')[1].strip() == '0':
    #             f_positive.write(author.split('Author:')[1] + '\n')
    #
    # with open('targets/negative_targets.txt', 'w', encoding='utf8') as f_negative:
    #     for author, line in test_set_tuple.items():
    #         if line.split('$LABEL$')[1].strip() == '1':
    #             f_negative.write(author.split('Author:')[1] + '\n')
    #
    # with open('targets/all_targets.txt', 'w', encoding='utf8') as f_all:
    #     for author, line in test_set_tuple.items():
    #         f_all.write(author.split('Author:')[1].strip() + '\n')

    preds, targets = predict_dataset(ckpt='fake_news_victim_model', pred_news_tuple=test_set_tuple)

    overall_disparity = calculate_overall_disparity(preds, targets)

    print(overall_disparity)
