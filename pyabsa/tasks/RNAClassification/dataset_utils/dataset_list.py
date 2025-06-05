# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:42



class RNACDatasetList(list):
    """
    A list of available RNA datasets.
    """

    def __init__(self):
        super(RNACDatasetList, self).__init__(self.__class__.__dict__.values())


class RNAClassificationDatasetList(RNACDatasetList):
    pass
