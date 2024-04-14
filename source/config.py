# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/12/8 21:09
# @file: config.py

import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            print(root_path)
            
            if 'source' in root_path.split('/'):
                self.base_path = os.path.abspath(os.path.join(os.path.pardir))
            else:
                self.base_path = os.path.abspath(os.path.join(os.getcwd(),'embedding','transX'))
            
            self._init_train_config()

    def __new__(cls, *args, **kwargs):
        """
        单例类-结合线程锁的机制，实现了单例模式，确保了在整个程序运行期间只有一个 Config 类的实例存在
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(Config, '_instance'):
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def _init_train_config(self):
        self.model_name = "TransD"
        self.data='data'
        self.outout='output'
        self.getTxt=True

        self.train_data_path = os.path.join(self.base_path, self.data, 'train.txt')
        self.eval_data_path = os.path.join(self.base_path, self.data, 'valid.txt')
        self.entity2id_path = os.path.join(self.base_path, self.data, 'entity2id.txt')
        self.relation2id_path = os.path.join(self.base_path, self.data, 'relation2id.txt')
        self.output_path = os.path.join(self.base_path, self.outout, self.model_name)
        self.logs_path = os.path.join(self.base_path, self.outout, 'logs')
        self.dump_embedding_path = os.path.join(self.output_path, 'embed') #嵌入向量保存的路径
        self.model_path = os.path.join(self.output_path, 'model')
        self.annoy_index_path = os.path.join(self.output_path, 'annoy') #Annoy 索引文件保存的路径

        self.do_train = True
        self.do_eval = True
        self.use_gpu = True

        # Dataloader arguments
        self.batch_size = 1024 #批次大小
        self.rep_proba = 0.5 #用于生成负样本时替换尾实体的概率
        self.ex_proba = 0.5 #用于生成负样本时替换关系的概率

        # Model and training general arguments
        self.TransE = {"EmbeddingDim": 100, "Margin": 1.0}
        self.TransH = {"EmbeddingDim": 100, "Margin": 1.0, "C": 0.01, "Eps": 0.001}
        self.TransD = {"EntityDim": 100, "RelationDim": 100, "Margin": 2.0}
        self.TransA = {"EmbeddingDim": 100, "Margin": 3.2, "Lamb": 0.01, "C": 0.2}
        self.KG2E = {"EmbeddingDim": 100, "Margin": 4.0, "Sim": "EL", "Vmin": 0.03, "Vmax": 3.0}
        self.weight_decay = 0 #权重衰减
        self.epochs = 5 #原来是5
        self.seed = 1 #原来是3
        self.eval_epoch = 1 #每隔多少轮进行一次评估
        self.learning_rate = 0.01 #学习率
        self.lr_decay = 0.96 #学习率衰减因子
        self.lr_decay_epoch = 5 #学习率衰减轮数
        self.eval_method = "MR" #评估方法，这里设置为 "MR"（Mean Rank）
        self.sim_measure = "L2" #相似度度量方式，这里设置为 "L2"（L2 范数）
        self.eval_batch_size = 1

       
