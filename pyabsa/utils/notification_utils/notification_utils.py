# -*- coding: utf-8 -*-
# file: notification_utils.py
# time: 05/11/2022 23:38

import os

from pyabsa.framework.flag_class.flag_template import PyABSAMaterialHostAddress

import requests
from termcolor import colored
from pyabsa import __version__ as pyabsa_version
from pyabsa.utils.pyabsa_utils import fprint


def check_emergency_notification():
    """
    Check if there is any emergency notification from PyABSA
    """

    url = PyABSAMaterialHostAddress + "resolve/main/emergency_notification.txt"

    try:  # from Huggingface Space
        response = requests.get(url, stream=True)
        save_path = "notification.txt"
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        with open(save_path, "r") as f:
            fprint(colored("PyABSA({}): ".format(pyabsa_version) + f.read(), "red"))
        os.remove(save_path)
    except Exception as e:
        pass
