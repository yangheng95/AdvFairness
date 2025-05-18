# -*- coding: utf-8 -*-
# file: rnar_configuration.py
# time: 02/11/2022 19:59
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import copy

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
from pyabsa.framework.configuration_class.configuration_template import ConfigManager
from ..models.__classic__.lstm import LSTM
from ..models.__plm__.bert import BERT_MLP

_rnar_config_template = {
    "model": BERT_MLP,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "patience": 99999,
    "cache_dataset": True,
    "warmup_step": -1,
    "show_metric": False,
    "max_seq_len": 80,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "evaluate_begin": 0,
    "cross_validate_fold": -1,
    "use_amp": False,
    "overwrite_cache": False,
    # split train and test datasets into 5 folds and repeat 3 trainer
}

_rnar_config_base = {
    "model": BERT_MLP,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "pretrained_bert": "microsoft/deberta-v3-base",
    "cache_dataset": True,
    "warmup_step": -1,
    "show_metric": False,
    "max_seq_len": 80,
    "patience": 99999,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "evaluate_begin": 0,
    "cross_validate_fold": -1
    # split train and test datasets into 5 folds and repeat 3 trainer
}

_rnar_config_english = {
    "model": BERT_MLP,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "patience": 99999,
    "pretrained_bert": "microsoft/deberta-v3-base",
    "cache_dataset": True,
    "warmup_step": -1,
    "show_metric": False,
    "max_seq_len": 80,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "evaluate_begin": 0,
    "cross_validate_fold": -1
    # split train and test datasets into 5 folds and repeat 3 trainer
}

_rnar_config_multilingual = {
    "model": BERT_MLP,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "patience": 99999,
    "pretrained_bert": "microsoft/mdeberta-v3-base",
    "cache_dataset": True,
    "warmup_step": -1,
    "show_metric": False,
    "max_seq_len": 80,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "evaluate_begin": 0,
    "cross_validate_fold": -1
    # split train and test datasets into 5 folds and repeat 3 trainer
}

_rnar_config_chinese = {
    "model": BERT_MLP,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "patience": 99999,
    "cache_dataset": True,
    "warmup_step": -1,
    "show_metric": False,
    "pretrained_bert": "bert-base-chinese",
    "max_seq_len": 80,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "evaluate_begin": 0,
    "cross_validate_fold": -1
    # split train and test datasets into 5 folds and repeat 3 trainer
}

_rnar_config_glove = {
    "model": LSTM,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "cache_dataset": True,
    "warmup_step": -1,
    "show_metric": False,
    "max_seq_len": 100,
    "patience": 20,
    "dropout": 0.1,
    "l2reg": 0.000001,
    "num_epoch": 100,
    "batch_size": 64,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "embed_dim": 300,
    "hidden_dim": 300,
    "output_dim": 3,
    "log_step": 5,
    "warm_step": -1,
    "hops": 3,  # valid in MemNet and RAM only
    "evaluate_begin": 0,
    "cross_validate_fold": -1,
}


class RNARConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:  {'model': MLP,
                            'optimizer': "adamw",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "roberta-base",
                            'cache_dataset': True,
                            'warmup_step': -1,
                            'show_metric': False,
                            'max_seq_len': 80,
                            'patience': 99999,
                            'dropout': 0,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {52, 25}
                            'embed_dim': 768,
                            'hidden_dim': 768,
                            'output_dim': 3,
                            'log_step': 10,
                            'evaluate_begin': 0,
                            'cross_validate_fold': -1 # split train and test datasets into 5 folds and repeat 3 trainer
                            }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def set_rnar_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == "template":
                _rnar_config_template.update(newitem)
            elif configType == "base":
                _rnar_config_base.update(newitem)
            elif configType == "english":
                _rnar_config_english.update(newitem)
            elif configType == "chinese":
                _rnar_config_chinese.update(newitem)
            elif configType == "multilingual":
                _rnar_config_multilingual.update(newitem)
            elif configType == "glove":
                _rnar_config_glove.update(newitem)
            else:
                raise ValueError(
                    "Wrong value of configuration_class type supplied, please use one from following type: template, base, english, chinese, multilingual, glove"
                )
        else:
            raise TypeError(
                "Wrong type of new configuration_class item supplied, please use dict e.g.{'NewConfig': NewValue}"
            )

    @staticmethod
    def set_rnar_config_template(newitem):
        RNARConfigManager.set_rnar_config("template", newitem)

    @staticmethod
    def set_rnar_config_base(newitem):
        RNARConfigManager.set_rnar_config("base", newitem)

    @staticmethod
    def set_rnar_config_english(newitem):
        RNARConfigManager.set_rnar_config("english", newitem)

    @staticmethod
    def set_rnar_config_chinese(newitem):
        RNARConfigManager.set_rnar_config("chinese", newitem)

    @staticmethod
    def set_rnar_config_multilingual(newitem):
        RNARConfigManager.set_rnar_config("multilingual", newitem)

    @staticmethod
    def set_rnar_config_glove(newitem):
        RNARConfigManager.set_rnar_config("glove", newitem)

    @staticmethod
    def get_rnar_config_template():
        _rnar_config_template.update(_rnar_config_template)
        return RNARConfigManager(copy.deepcopy(_rnar_config_template))

    @staticmethod
    def get_rnar_config_base():
        _rnar_config_template.update(_rnar_config_base)
        return RNARConfigManager(copy.deepcopy(_rnar_config_template))

    @staticmethod
    def get_rnar_config_english():
        _rnar_config_template.update(_rnar_config_english)
        return RNARConfigManager(copy.deepcopy(_rnar_config_template))

    @staticmethod
    def get_rnar_config_chinese():
        _rnar_config_template.update(_rnar_config_chinese)
        return RNARConfigManager(copy.deepcopy(_rnar_config_template))

    @staticmethod
    def get_rnar_config_multilingual():
        _rnar_config_template.update(_rnar_config_multilingual)
        return RNARConfigManager(copy.deepcopy(_rnar_config_template))

    @staticmethod
    def get_rnar_config_glove():
        _rnar_config_template.update(_rnar_config_glove)
        return RNARConfigManager(copy.deepcopy(_rnar_config_template))
