# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:45



class TADDatasetList(list):
    """
    Classification Datasets for adversarial attack defense
    """

    def __init__(self):
        super(TADDatasetList, self).__init__(self.__class__.__dict__.values())
