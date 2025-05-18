# -*- coding: utf-8 -*-
# file: text_attack.py
# time: 22:24 07/11/2023
# target: YANG, HENG <hy345@exeter.ac.uk> 
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import random
import tqdm
import os
import pickle

from textattack import Attacker
from textattack.attack_recipes import BERTAttackLi2020, BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, \
    IGAWang2019, GeneticAlgorithmAlzantot2018, DeepWordBugGao2018, CLARE2020
from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult, FailedAttackResult
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper

import pyabsa.tasks.FairnessAttackClassification as TC

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


class PyABSAModelWrapper(HuggingFaceModelWrapper):
    """ Transformers sentiment analysis pipeline returns a list of responses
        like

            [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

        We need to convert that to a format TextAttack understands, like

            [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model):
        self.model = model  # pipeline = pipeline

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.predict(text_input, print_result=False, ignore_error=False, **kwargs)
            outputs.append(raw_outputs['probs'])
        return outputs


class TextAttacker:

    def __init__(self, model, recipe_class=BAEGarg2019):
        model = model
        model_wrapper = PyABSAModelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')

        # recipe.transformation.language = "en"

        _dataset = [('', 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


if __name__ == '__main__':
    # dataset_name = 'news_sentiment'
    # dataset_name = 'app_reviews'
    dataset_name = 'fake_news'

    # victim_model = TC.TextClassifier('victim_model')
    victim_model = TC.TextClassifier('bert_mlp')

    adversarial_attacks = {
        # 'BERTAttackLi2020': BERTAttackLi2020,
        # 'CLARE2020': CLARE2020,
        # 'DeepWordBugGao2018': DeepWordBugGao2018,
        # 'GeneticAlgorithmAlzantot2018': GeneticAlgorithmAlzantot2018,
        # 'IGAWang2019': IGAWang2019,
        # 'PSOZang2020': PSOZang2020,
        # 'BAEGarg2019': BAEGarg2019,
        # 'PWWSRen2019': PWWSRen2019,
        'TextFoolerJin2019': TextFoolerJin2019
    }

    for attack_name, attack_class in adversarial_attacks.items():
        # text_attacker = TextAttacker(victim_model, recipe_class=TextFoolerJin2019)
        text_attacker = TextAttacker(victim_model, recipe_class=attack_class)

        test_dataset_tuple = prepare_test_dataset(dataset_name=dataset_name)

        if os.path.exists('{}_{}_attacked.pkl'.format(dataset_name, attack_name)):
            attacked_dataset_tuple = pickle.load(open('{}_{}_attacked.pkl'.format(dataset_name, attack_name), 'rb'))
        else:
            attacked_dataset_tuple = []
            overall_disparity = []
            for target, text in tqdm.tqdm(test_dataset_tuple):
                attacked_result = text_attacker.attacker.simple_attack(text, 0 if text.split('$LABEL$')[1].strip()=='0' else 1)
                attacked_text = attacked_result.perturbed_result.attacked_text.text
                attacked_label = attacked_result.perturbed_result.output
                if isinstance(attacked_result, SuccessfulAttackResult):
                    attacked_dataset_tuple.append((target, target+'\t'+'\t'.join(attacked_text.split('\t')[1:-1])+'$LABEL${}'.format(attacked_label)))
                else:
                    attacked_dataset_tuple.append((target, text))
            with open('{}_{}_attacked.pkl'.format(dataset_name, attack_name), 'wb') as f:
                pickle.dump(attacked_dataset_tuple, f)

        # preds, targets = predict_dataset(victim_model, pred_news_tuple=attacked_dataset_tuple)


        print('-' * 100)
        print('Clean Dataset Fairness and Performance (BinaryLabelDatasetMetric):')
        clean_results = evaluate_performance(victim_model, pred_news_tuple=test_dataset_tuple)
        true_labels = []
        pred_labels = []
        targets = []
        for result, (target, text) in zip(clean_results, test_dataset_tuple):
            targets.append(target)
            pred_labels.append(result['label'])
            true_labels.append(text.split('$LABEL$')[1].strip())

        print('Clean Dataset Fairness and Performance (BinaryLabelDatasetMetric):')
        overall_disparity = calculate_binary_dataset_disparity(true_labels, protected_attributes=targets,
                                                               dataset_name=dataset_name)
        print('Disparate Impact: {}'.format(overall_disparity['disparate_impact']))
        print('Statistical Parity Difference: {}'.format(overall_disparity['statistical_parity_difference']))
        print(classification_report(true_labels, pred_labels))

        print('-' * 100)
        print('Attacked Dataset Fairness and Performance (BinaryLabelDatasetMetric):')

        attacked_results = evaluate_performance(victim_model, pred_news_tuple=attacked_dataset_tuple)

        true_labels = []
        pred_labels = []
        targets = []
        for result, (target, text) in zip(attacked_results, test_dataset_tuple):
            targets.append(target)
            pred_labels.append(result['label'])
            true_labels.append(text.split('$LABEL$')[1].strip())

        print('Attacked Dataset Fairness and Performance (BinaryLabelDatasetMetric):')
        overall_disparity = calculate_binary_dataset_disparity(pred_labels, protected_attributes=targets,
                                                               dataset_name=dataset_name)
        print('Disparate Impact: {}'.format(overall_disparity['disparate_impact']))
        print('Statistical Parity Difference: {}'.format(overall_disparity['statistical_parity_difference']))
        print(classification_report(true_labels, pred_labels))

        print('-' * 100)

        # print('Attacked Dataset Fairness and Performance (ClassificationMetric):')
        # overall_disparity = calculate_overall_disparity(true_labels, predicted_labels=attacked_preds,
        #                                                 protected_attributes=targets, dataset_name=dataset_name)
        # print('Disparate Impact: {}'.format(overall_disparity['disparate_impact']))
        # print('Statistical Parity Difference: {}'.format(overall_disparity['statistical_parity_difference']))
        #
