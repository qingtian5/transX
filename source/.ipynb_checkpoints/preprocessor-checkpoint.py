# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: preprocessor.py.py

import os
from collections import Counter

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import Config
from logger import logger
from utils import load_file


class TransXProcessor(object):

    @staticmethod
    def generate_data(config: Config):
        """
        根据数据集生成entity2id,relation2id文件，此操作可以通过其他方式实现，当数据量过大时，可以采用hadoop进行处理
        :param config:
        :return:
        """
        data_path = []
        if config.do_train:
            data_path.append(config.train_data_path)

        if config.do_eval:
            data_path.append(config.eval_data_path)

        raw_df = pd.concat([pd.read_csv(p,
                                        sep="\t",
                                        header=None,
                                        names=["head", "relation", "tail"],
                                        keep_default_na=False,
                                        encoding="utf-8") for p in data_path], axis=0)
        raw_df.reset_index(drop=True, inplace=True)

        head_counter = Counter(raw_df["head"])
        tail_counter = Counter(raw_df["tail"])
        relation_counter = Counter(raw_df["relation"])

        # Generate entity and relation list
        entity_list = list((head_counter + tail_counter).keys())
        relation_list = list(relation_counter.keys())

        # Transform to index dict and save
        entity_index = [[word, index] for index, word in enumerate(entity_list)]
        relation_index = [[word, index] for index, word in enumerate(relation_list)]

        pd.DataFrame(entity_index).to_csv(config.entity2id_path, sep="\t", header=False, index=False, encoding="utf-8")
        pd.DataFrame(relation_index).to_csv(config.relation2id_path, sep="\t", header=False, index=False,
                                            encoding="utf-8")

    def init_data_dict(self, config: Config):
        """
        初始化实体和实体ID、关系和关系ID
        :param config:
        :return:
        """
        # 用来存放实体和实体ID、关系和关系ID
        # 格式: {关系:关系ID}、{关系:关系ID}
        entity_dict, entity_total = self._read_ent_rel_data(config.entity2id_path)
        relation_dict, relation_total = self._read_ent_rel_data(config.relation2id_path)
        return entity_dict, relation_dict

    @staticmethod
    def _read_ent_rel_data(path):
        content_dict = dict()
        content_len = 0
        try:
            lines = load_file(path)
            for i in lines:
                content_dict[i.strip().split('\t')[0]] = int(i.strip().split('\t')[1])
            content_len = len(lines)
        except IOError:
            logger.error("Error: 没有找到文件或读取文件失败", path)
            exit(-1)
        return content_dict, content_len

    @staticmethod
    def data_set(config: Config, entity_dict, relation_dict, mode="train"):
        """
        获取指定数据集，将其转换成ID
        :param config:
        :param entity_dict:
        :param relation_dict:
        :param mode:
        :return:
        """
        # 用来存放三元组数据,格式:头实体\t关系\t尾实体
        triple_lists = list()
        triple_path = config.train_data_path if mode == "train" else config.eval_data_path

        def word2id(word):
            try:
                return int(entity_dict[word])
            except KeyError:
                return int(relation_dict[word])

        try:
            lines = load_file(triple_path)
            for i in lines:
                triple_list = i.split('\t')
                head, relation, tail = triple_list[0].strip(), triple_list[1].strip(), triple_list[2].strip()
                triple_lists.append(
                    str(word2id(head)) + "_" + str(word2id(relation)) + "_" + str(word2id(tail)))
        except IOError:
            logger.error('文件夹内没有triple2id.txt，\ntriple2id.txt文件每一行是:开始实体\\t结束实体\\t关系')
            exit(-1)

        return triple_lists

    @staticmethod
    def data_loader(config, data_set, mode="train"):
        """
        获取实体、关系、data_loader
        :param config:
        :param data_set:
        :param mode:
        :return:
        """
        assert (mode in ["train", "eval"])
        tds = TransXDataSet(data_set=data_set)
        if mode == "train":
            # 生成负样本
            tds.generate_neg_sampler(rep_prob=config.rep_proba, ex_prob=config.ex_proba)
            loader = DataLoader(tds, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=1024)
        else:
            loader = DataLoader(tds, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=1024)
        return loader

    @staticmethod
    def clean_output(config: Config):
        """
        清理output目录，若output目录存在，将会被删除, 然后初始化输出目录
        :param config:
        :return:
        """
        if config.do_train:
            logger.info(f"check up output dir and clear dir: {config.output_path}")
            if os.path.exists(config.output_path):
                def del_file(path):
                    ls = os.listdir(path)
                    for i in ls:
                        c_path = os.path.join(path, i)
                        if os.path.isdir(c_path):
                            del_file(c_path)
                            os.rmdir(c_path)
                        else:
                            os.remove(c_path)

                try:
                    del_file(config.output_path)
                except Exception as e:
                    logger.error(e)
                    logger.error('pleace remove the files of output dir and data.conf')
                    exit(-1)

        # 初始化output目录
        if os.path.exists(config.output_path) and os.listdir(config.output_path) and config.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(config.output_path))

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        if not os.path.exists(config.dump_embedding_path):
            os.makedirs(config.dump_embedding_path)

        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)


class TransXDataSet(Dataset):
    def __init__(self, data_set):
        super(Dataset, self).__init__()
        self.df = self._list2data_frame(data_set=data_set)

    @staticmethod
    def _list2data_frame(data_set):
        res = [i.split("_") for i in data_set]
        df = pd.DataFrame(res, columns=["head", "relation", "tail"])
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if hasattr(self, "neg_df"):
            return np.array(self.df.iloc[item, :3], dtype=np.int64), np.array(self.neg_df.iloc[item, :3],
                                                                              dtype=np.int64)
        else:
            return np.array(self.df.iloc[item, :3], dtype=np.int64)

    def generate_neg_sampler(self, rep_prob=0.5, ex_prob=0.5):
        """
        假设将所有输入正例三元组看成是shape(N, 3)的矩阵，按照论文的要求，要么替换head，要么替换tail.

        shuffle_head：将矩阵第一列使用random.shuffle()得到的混排的head(相当于论文里面所说的随机找一个替换head的实体)
        shuffle_tail：将矩阵第三列使用random.shuffle()得到的混排的tail(相当于论文里面所说的随机找一个替换tail的实体)
        rep_prob_distribution：使用random.random((0, 1))生成长度为N的随机数列表
        ex_prob_distribution：使用random.random((0, 1))生成长度为N的随机数列表

        那么可以设定一个阈值rep_proba，如果rep_prob_distribution的值小于rep_proba，那么使用shuffle_head对应位置的实体替换原来的头实体，
        反之，使用shuffle_tail对应位置的实体替换原来的尾实体。

        可以看出，当N足够大时，产生出来的负样本与正样本碰撞的概率几乎为零，基本符合论文要求。
        :param rep_prob: (float)Probability of replacing head
        :param ex_prob:  (float)Probability of replacing head with tail entities or replacing tail with head entities.
        :return:
        """
        self.neg_df = self.df.copy()

        shuffle_head = self.neg_df["head"].sample(frac=1.0, random_state=0)
        shuffle_tail = self.neg_df["tail"].sample(frac=1.0, random_state=0)

        np.random.seed(0)
        rep_prob_distribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.neg_df),))

        np.random.seed(0)
        ex_prob_distribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.neg_df),))

        # Replacing head or tail
        def replace_head(rel_head, shuffle_head, shuffle_tail, rep_p, ex_p):
            if rep_p >= rep_prob:
                '''
                Not replacing head.self.negD
                '''
                return rel_head
            else:
                if ex_p > ex_prob:
                    '''
                    Replacing head with shuffle head.
                    '''
                    return shuffle_head
                else:
                    '''
                    Replacing head with shuffle tail.
                    '''
                    return shuffle_tail

        def replace_tail(rel_tail, shuffle_head, shuffle_tail, rep_p, ex_p):
            if rep_p < rep_prob:
                '''
                Not replacing tail.
                '''
                return rel_tail
            else:
                if ex_p > ex_prob:
                    '''
                    Replacing tail with shuffle tail.
                    '''
                    return shuffle_tail
                else:
                    '''
                    Replacing head with shuffle head.
                    '''
                    return shuffle_head

        self.neg_df["head"] = list(
            map(replace_head, self.neg_df["head"], shuffle_head, shuffle_tail, rep_prob_distribution,
                ex_prob_distribution))
        self.neg_df["tail"] = list(
            map(replace_tail, self.neg_df["tail"], shuffle_head, shuffle_tail, rep_prob_distribution,
                ex_prob_distribution))


if __name__ == '__main__':
    TransXProcessor().generate_data(config=Config())
