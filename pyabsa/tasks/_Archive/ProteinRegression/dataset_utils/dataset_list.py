# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:43



class ProteinRDatasetList(list):
    """
    Protein Sequence-based Regression Dataset Lists
    """

    def __init__(self):
        super(ProteinRDatasetList, self).__init__(self.__class__.__dict__.values())


class ProteinRegressionDatasetList(list):
    """
    Protein Sequence-based Regression Dataset Lists
    """

    def __init__(self):
        super(ProteinRegressionDatasetList, self).__init__(
            self.__class__.__dict__.values()
        )
