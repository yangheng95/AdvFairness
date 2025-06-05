# -*- coding: utf-8 -*-

from abc import ABC

from textattack.shared.utils import ReprMixin


class ReactiveDefender(ReprMixin, ABC):
    def __init__(self, **kwargs):
        pass

    def warn_adversary(self, **kwargs):
        pass

    def repair(self, **kwargs):
        pass
