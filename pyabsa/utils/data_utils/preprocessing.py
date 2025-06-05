﻿# -*- coding: utf-8 -*-
# file: preprocessing.py
# time: 19:28 2023/3/1


import json

import pandas


def parse_data_tuple(data_instance):
    """
    Parse text to dict
    :param data_instance: text to be parsed
    :return:

    """

    if isinstance(data_instance, dict):
        return data_instance

    elif isinstance(data_instance, str):
        try:
            return json.loads(data_instance)
        except Exception as e:
            return data_instance

    elif isinstance(data_instance, pandas.DataFrame):
        return data_instance.to_tuple(orient="records")
