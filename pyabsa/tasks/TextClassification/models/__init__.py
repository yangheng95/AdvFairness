# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 15:47



class BERTTCModelList(list):
    from .__plm__.bert import BERT_MLP

    BERT_MLP = BERT_MLP
    BERT = BERT_MLP

    def __init__(self):
        super(BERTTCModelList, self).__init__([self.BERT_MLP])


class GloVeTCModelList(list):
    from .__classic__.lstm import LSTM

    LSTM = LSTM

    def __init__(self):
        super(GloVeTCModelList, self).__init__([self.LSTM])
