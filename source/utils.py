# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: utils.py

import csv
import hashlib
import json
import os
import pickle
import time
from datetime import timedelta, datetime
from glob import iglob
from mmap import mmap, ACCESS_READ

import numpy as np
import yaml

'''这个代码是一个工具模块
    其中定义了一系列用于文件操作、数据处理、时间处理等功能的函数
    '''
def load_big_file(fp: str):
    """
    读取大文件
    :param fp:
    1、使用 mmap() 函数将文件映射到内存中，这样可以在不加载整个文件到内存中的情况下对文件进行操作
        文件句柄 f.fileno()
        映射的长度 0（表示映射整个文件）
        访问权限 ACCESS_READ（表示只读）
    2、enumerate() 函数遍历内存映射对象 m，每次迭代产生一个字符的索引和对应的字符
        当遇到换行符 "\n" 时，说明找到了一行的结束位置，此时利用切片操作 m[tmp:i + 1] 获取从上次记录的位置到当前位置的字节序列
        调用 decode() 方法将其解码为字符串
    :return:  <class 'generator'>
    """
    with open(fp, "r", encoding="utf-8") as f:
        m = mmap(f.fileno(), 0, access=ACCESS_READ)
        tmp = 0
        for i, char in enumerate(m):
            if char == b"\n":
                yield m[tmp:i + 1].decode()
                tmp = i + 1


def load_file(fp: str, sep: str = None, name_tuple=None):
    """
    读取文件；
    若sep为None，按行读取，返回文件内容列表，格式为:[xxx,xxx,xxx,...]
    若不为None，按行读取分隔，返回文件内容列表，格式为: [[xxx,xxx],[xxx,xxx],...]
    :param fp:
    :param sep:
    :param name_tuple:
    :return:
    """
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple: 
                return map(name_tuple._make, [line.strip().split(sep) for line in lines]) 
            #命名元组 _make 方法将其转换为命名元组
            #map() 函数将所有命名元组组成的迭代器作为返回值
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines


def load_json_file(fp: str):
    """
    加载json文件
    :param fp:
    :return:
    """
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(line.strip(), encoding="utf-8") for line in f.readlines()]


def save_json_file(list_data, fp):
    """
    保存json文件
    :param list_data:
    :param fp:
    :return:
    """
    with open(fp, "w", encoding="utf-8") as f:
        for data in list_data:
            '''对每个元素调用 json.dumps() 函数将其转换为 JSON 格式的字符串ensure_ascii=False 参数表示允许非 ASCII 字符的输出'''
            json_str = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(json_str))
        f.flush() #调用 flush() 方法刷新文件缓冲区，确保数据被写入文件


def load_csv(fp, is_tsv: bool = False):
    """
    加载csv文件为OrderDict()列表
    :param fp:
    :param is_tsv:
    :return:
    """
    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(fp, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def load_csv_tuple(fp, name_tuple):
    """
    加载csv文件为指定类
    :param fp:
    :param name_tuple:
    :return:
    """
    return map(name_tuple._make, csv.reader(open(fp, "r", encoding="utf-8")))


def load_pkl(fp):
    """
    加载pkl文件
    :param fp:
    :return:
    """
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp):
    """
    保存pkl文件，数据序列化
    :param data:
    :param fp:
    :return:
    """
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_yaml(fp):
    """
    加载yml文件
    :param fp:
    :return:
    """
    return yaml.load(open(fp, "r", encoding='utf-8'),
                     Loader=yaml.SafeLoader)


def save_yaml(data, fp):
    """
    持久化yml文件
    :param data:
    :param fp:
    :return:
    """
    yaml.dump(
        data,
        open(fp, "w", encoding="utf-8"),
        allow_unicode=True,
        default_flow_style=False)


def word2id(c2id: dict, word: str):
    """
    将一个词用词典c2id组织成列表的形式

    :param c2id: 字符->id的词典
    :param word: 输入的单词
    :return:    输入单词对应的字符索引列表
    """
    word_id = list()

    for i, c in enumerate(word):
        if c in c2id:
            word_id.append(c2id[c])
        else:
            word_id.append(0)
    return word_id


def id2word(id2c: dict, ids: list):
    """
    将一个字符索引列表组织成单词

    :param id2c: id->字符的词典
    :param ids:  输入的字符索引列表
    :return:     字符列表对应的单词
    """
    word = ""
    for i in ids:
        if i in id2c:
            word += id2c[i]
        else:
            return ""
    return word


def calculate_distance(vector1, vector2):
    """
    计算两个向量的余弦相似度
    :param vector1: 向量1
    :param vector2: 向量2
    :return:
    """
    cosine_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))  # 余弦夹角
    # euclidean_distance = np.sqrt(np.sum(np.square(vector1 - vector2)))  # 欧式距离
    return cosine_distance


def split_data(data_set, batch_size):
    """
    数据集切分
    :param data_set:
    :param batch_size:
    :return:
    """
    batch_list = []
    data_size = len(data_set)
    count = (data_size + batch_size - 1) // batch_size
    for i in range(count):
        last_index = data_size if (i + 1) * batch_size > data_size else (i + 1) * batch_size
        res = data_set[i * batch_size:last_index]
        batch_list.append(res)
    return batch_list


def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time: time.time()
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def format_data(t: datetime):
    """
    时间格式化，time.strftime("%Y-%m-%d %H:%M:%S")
    :param t:
    :return:
    """
    return t.strftime("%Y-%m-%d %H:%M:%S")


def md5(s: str):
    """
    MD5加密
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()


def scan_fp(ph):
    """
    递归返回指定目录下的所有文件
    :param ph:
    :return:
    """
    path_list = []
    for p in os.listdir(ph):
        fp = os.path.join(ph, p)
        if os.path.isfile(fp):
            path_list.append(fp)
        elif os.path.isdir(fp):
            path_list.extend(scan_fp(fp))
    return path_list


def scan_fp_iglob(ph):
    """
    返回指定目录下的所有文件
    :param ph:
    :return:
    """
    return list(filter(lambda x: os.path.isfile(x), iglob(f"{ph}/**", recursive=True)))
