# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 15:47



class ASTEModelList(list):
    from .model import EMCGCN

    EMCGCN = EMCGCN

    def __init__(self):
        super(ASTEModelList, self).__init__([self.EMCGCN])
