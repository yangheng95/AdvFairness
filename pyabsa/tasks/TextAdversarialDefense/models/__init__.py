# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 15:47



class BERTTADModelList(list):
    from .__plm__.tad_bert import TADBERT

    TADBERT = TADBERT

    def __init__(self):
        super(BERTTADModelList, self).__init__([self.TADBERT])


class GloVeTADModelList(list):
    from .__classic__.tad_lstm import TADLSTM

    TADLSTM = TADLSTM

    def __init__(self):
        super(GloVeTADModelList, self).__init__([self.TADLSTM])
