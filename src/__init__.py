# -*- coding: utf-8 -*-
# @Author     : lipf702@gmail.com
# @Since      : 12/10/23
# @Description:

import os


def _set_up_openai_api_key():
    """ 从本地文件读取openai api key设置环境变量 """
    config_path = "../config/openai_api_key.local"
    with open(config_path, "r") as fr:
        key = fr.read().strip()
    os.environ["OPENAI_API_KEY"] = key
