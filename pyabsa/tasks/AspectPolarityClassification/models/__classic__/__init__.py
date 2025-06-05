# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 15:48



class ClassicAPCModelList(list):
    from .aoa import AOA
    from .asgcn import ASGCN
    from .atae_lstm import ATAE_LSTM
    from .cabasc import Cabasc
    from .ian import IAN
    from .lstm import LSTM
    from .memnet import MemNet
    from .mgan import MGAN
    from .ram import RAM
    from .tc_lstm import TC_LSTM
    from .td_lstm import TD_LSTM
    from .tnet_lf import TNet_LF

    AOA = AOA
    ASGCN = ASGCN
    ATAE_LSTM = ATAE_LSTM
    Cabasc = Cabasc
    IAN = IAN
    LSTM = LSTM
    MemNet = MemNet
    MGAN = MGAN
    RAM = RAM
    TC_LSTM = TC_LSTM
    TD_LSTM = TD_LSTM
    TNet_LF = TNet_LF

    def __init__(self):
        super(ClassicAPCModelList, self).__init__(
            [
                self.ASGCN,
                self.AOA,
                self.ATAE_LSTM,
                self.Cabasc,
                self.IAN,
                self.LSTM,
                self.MemNet,
                self.MGAN,
                self.RAM,
                self.TC_LSTM,
                self.TD_LSTM,
                self.TNet_LF,
            ]
        )


class GloVeAPCModelList(ClassicAPCModelList):
    pass
