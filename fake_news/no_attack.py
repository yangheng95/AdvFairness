# -*- coding: utf-8 -*-
# file: no_attack.py
# time: 18:50 06/11/2023
# author: YANG, HENG 
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import sklearn.metrics
from sklearn.metrics import classification_report

from fairness_utils import (
    prepare_test_dataset,
    calculate_overall_disparity,
    calculate_binary_dataset_disparity,
    predict_dataset,
    load_all_targets,
    load_negative_targets,
    load_positive_targets,
    evaluate_performance
)
import random
import pyabsa.tasks.FairnessAttackClassification as TC

if __name__ == '__main__':

    dataset_name = 'fake_news'
    victim_model = TC.TextClassifier('victim_model')

    overall_disparity = []
    for seed in [1]:
        random.seed(seed)

        all_targets = load_all_targets(dataset_name=dataset_name)

        test_dataset_tuple = prepare_test_dataset(dataset_name=dataset_name)

        preds, targets = predict_dataset(victim_model, pred_news_tuple=test_dataset_tuple)

        results = evaluate_performance(victim_model, pred_news_tuple=test_dataset_tuple)

        true_labels = []
        pred_labels = []
        for result, (target, text) in zip(results, test_dataset_tuple):
            true_labels.append(text.split('$LABEL$')[1].strip())
            pred_labels.append(result['label'])

        print(classification_report(true_labels, pred_labels, digits=4))

        overall_disparity.append(calculate_binary_dataset_disparity(preds, protected_attributes=targets, dataset_name=dataset_name))

    print('Mean Difference: {}'.format(
        sum([item['mean_difference'] for item in overall_disparity]) / len(overall_disparity)))
    print('Disparate Impact: {}'.format(
        sum([item['disparate_impact'] for item in overall_disparity]) / len(overall_disparity)))
    print('Statistical Parity Difference: {}'.format(
        sum([item['statistical_parity_difference'] for item in overall_disparity]) / len(overall_disparity)))
