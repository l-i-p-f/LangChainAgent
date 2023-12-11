# -*- coding: utf-8 -*-
# @Author     : lipf702@gmail.com
# @Since      : 12/11/23
# @Description:

import yaml

CONFIG_PATH = "../config/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as fr:
        return yaml.safe_load(fr)
