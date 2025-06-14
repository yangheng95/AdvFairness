# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:41
 

from pyabsa.utils.data_utils.dataset_item import DatasetItem


class CDDDatasetList(list):
    """
    Text Classification or Sentiment analysis datasets
    """

    Promise = DatasetItem("Promise", "401.Promise")
    GHPR = DatasetItem("GHPR", "402.GHPR")
    Devign = DatasetItem("Devign", "403.Devign")

    def __init__(self):
        super(CDDDatasetList, self).__init__(
            [
                self.Promise,
                self.GHPR,
                self.Devign,
            ]
        )


class CodeDefectDetectionDatasetList(CDDDatasetList):
    pass
