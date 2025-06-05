# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:43
 


class RNARDatasetList(list):
    """
    RNA Sequence-based Regression Dataset Lists
    """

    def __init__(self):
        super(RNARDatasetList, self).__init__(self.__class__.__dict__.values())


class RNARegressionDatasetList(list):
    """
    RNA Sequence-based Regression Dataset Lists
    """

    def __init__(self):
        super(RNARegressionDatasetList, self).__init__(self.__class__.__dict__.values())
