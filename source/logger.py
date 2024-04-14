# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: logger.py

import logging
import os

from config import Config

config = Config()

if not os.path.exists(config.logs_path):
    os.makedirs(config.logs_path)

logger = logging.getLogger("client_log")
# Log等级总开关s
logger.setLevel(logging.INFO)

# 创建handler，用于输出到控制台、写入日志文件
stream_handler = logging.StreamHandler()
log_file_handler = logging.FileHandler(filename=os.path.join(config.logs_path, "transx.log"), encoding="utf-8")

# 定义handler的输出格式-使用了时间、文件名、行号、日志级别和消息文本作为日志信息的格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

stream_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(log_file_handler)
