# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/12/9 20:47
# @file: main.py

import json
import os
import joblib

import numpy as np
import torch
from annoy import AnnoyIndex
from torch.autograd import Variable
from tqdm import trange, tqdm

import evaluation
from config import Config
from logger import logger
from models.KG2E import KG2E
from models.TransA import TransA
from models.TransD import TransD
from models.TransE import TransE
from models.TransH import TransH
from preprocessor import TransXProcessor
from utils import save_json_file, calculate_distance


class TransX(object):
    def __init__(self):
        self.processor = TransXProcessor()
        self.config = Config()
        if self.config.getTxt:
            TransXProcessor.generate_data(config=self.config)
        self.entity_dict, self.relation_dict = self.processor.init_data_dict(config=self.config)
        self.id2entity = {v: k for k, v in self.entity_dict.items()}
        self.entity_num, self.relation_num = len(self.entity_dict.keys()), len(self.relation_dict.keys())

    def _init_model(self):
        """
        根据配置项model_name,初始化对应的模型
        :return:
        """

        if self.config.model_name == "TransE":
            model = TransE(entity_num=self.entity_num,
                           relation_num=self.relation_num,
                           embedding_dim=self.config.TransE["EmbeddingDim"],
                           margin=self.config.TransE["Margin"])
        elif self.config.model_name == "TransH":
            model = TransH(entity_num=self.entity_num,
                           relation_num=self.relation_num,
                           embedding_dim=self.config.TransH["EmbeddingDim"],
                           margin=self.config.TransH["Margin"],
                           C=self.config.TransH["C"],
                           eps=self.config.TransH["Eps"])
        elif self.config.model_name == "TransA":
            model = TransA(entity_num=self.entity_num,
                           relation_num=self.relation_num,
                           embedding_dim=self.config.TransA["EmbeddingDim"],
                           margin=self.config.TransA["Margin"],
                           lamb=self.config.TransA["Lamb"],
                           C=self.config.TransA["C"])
        elif self.config.model_name == "TransD":
            model = TransD(entity_num=self.entity_num,
                           relation_num=self.relation_num,
                           entity_dim=self.config.TransD["EntityDim"],
                           relation_dim=self.config.TransD["RelationDim"],
                           margin=self.config.TransD["Margin"])
        elif self.config.model_name == "KG2E":
            model = KG2E(entity_num=self.entity_num,
                         relation_num=self.relation_num,
                         embedding_dim=self.config.KG2E["EmbeddingDim"],
                         margin=self.config.KG2E["Margin"],
                         sim=self.config.KG2E["Sim"],
                         vmin=self.config.KG2E["Vmin"],
                         vmax=self.config.KG2E["Vmax"])
        else:
            model = None
            print("ERROR : No model named %s" % self.config.model_name)
            exit(1)

        return model

    @staticmethod
    def adjust_learning_rate(optimizer, decay):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay

    def train(self):
        """
        模型训练
        :return:
        """
        # 对输出目录进行初始化操作,当output存在,则删除
        self.processor.clean_output(self.config)

        use_gpu = self.config.use_gpu and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")

        logger.info("available device: {}，count_gpu: {}".format(device, torch.cuda.device_count()))

        # 获取eval数据集以及对应的data_loader
        eval_data_set = self.processor.data_set(config=self.config, entity_dict=self.entity_dict,
                                                relation_dict=self.relation_dict, mode="eval")
        eval_data_loader = self.processor.data_loader(config=self.config, data_set=eval_data_set, mode="eval")

        # 获取训练数据集
        train_data_set = self.processor.data_set(config=self.config, entity_dict=self.entity_dict,
                                                 relation_dict=self.relation_dict, mode="train")
        # 初始化模型
        model = self._init_model()
        if model:
            print(device)
            model.to(device)

        if self.config.do_train:
            min_loss, best_mr = float("inf"), float("inf")

            # 优化器
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.config.weight_decay,
                                         lr=self.config.learning_rate)

            logger.info("************** Running training ****************")
            logger.info("Num Examples = {}".format(len(train_data_set)))
            logger.info("Num Epochs = {}".format(int(self.config.epochs)))
            logger.info("Num Seed = {}".format(self.config.seed))

            for seed in trange(int(self.config.seed), desc="Seed"):
                train_data_loader = self.processor.data_loader(config=self.config, data_set=train_data_set,
                                                               mode="train")
                print(train_data_loader)
                for epoch in trange(int(self.config.epochs), desc="Epoch"):

                    for step, batch in enumerate(tqdm(train_data_loader, desc="Iteration")):
                        posX, negX = batch
                        if use_gpu:
                            posX = Variable(posX.to(device))
                            negX = Variable(negX.to(device))
                        else:
                            posX = Variable(posX)
                            negX = Variable(negX)

                        # Normalize the embedding if neccessary
                        model.normalize_embedding()

                        # Calculate the loss from the model
                        loss = model(posX, negX)
                        loss_val = loss.item()

                        # Calculate the gradient and step down
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Print infomation and add to summary
                        if min_loss > loss_val:
                            min_loss = loss_val

                    logger.info(f"Seed: {seed}, epoch: {epoch}, iteration is finished, the min_loss is {min_loss}")

                    if (epoch + 1) % self.config.lr_decay_epoch == 0:
                        self.adjust_learning_rate(optimizer, decay=self.config.lr_decay)

                    if self.config.do_eval and (epoch + 1) % self.config.eval_epoch == 0:
                        logger.info("********* Running eval start **********")
                        mr = evaluation.mr_evaluation(eval_loader=eval_data_loader, model=self.config.model_name,
                                                      sim_measure=self.config.sim_measure,
                                                      **model.ret_eval_weights())
                        logger.info("running eval end... MR score is {}".format(mr))

                        if mr < best_mr:
                            logger.info(f"-------------the best_mr is {mr}, save model--------------")
                            best_mr = mr
                            torch.save(model, os.path.join(self.config.model_path, "kg_model.ckpt"))

            logger.info(f"<<<<<< the global_best_mr is {best_mr}, the min_loss is {min_loss} >>>>>>")
            logger.info("train the {} model successful!!! save dir is {}".format(self.config.model_name,
                                                                                 self.config.model_path))

            logger.info("start dump_embedding, save dir is {}".format(self.config.dump_embedding_path))
            self.dump_embedding(model)
            logger.info("dump_embedding successful!!!")

            #logger.info("start dump_annoy_index, save index dir is {}".format(self.config.annoy_index_path))
            #self.dump_annoy_index(model)
            #logger.info("dump_annoy_index successful!!!")

    def dump_embedding(self, load_model):
        """
        向量持久化
        :param load_model:
        :return:
        """
        ent_weight = load_model.entityEmbedding.weight.detach().cpu().numpy()
        rel_weight = load_model.relationEmbedding.weight.detach().cpu().numpy()
        entity_embedding_path = os.path.join(self.config.dump_embedding_path, "entity_embedding.txt")
        relation_embedding_path = os.path.join(self.config.dump_embedding_path, "relation_embedding.txt")
        # 保存实体向量
        ent_embed_list = [dict(word=ent, word_vet=embed) for ent, embed in
                          zip(self.entity_dict.keys(), ent_weight.tolist())]
        save_json_file(ent_embed_list, entity_embedding_path)
        # 保存关系向量
        rel_embed_list = [dict(word=rel, word_vet=embed) for rel, embed in
                          zip(self.relation_dict.keys(), rel_weight.tolist())]
        save_json_file(rel_embed_list, relation_embedding_path)

    def dump_annoy_index(self, load_model):
        """
        构建annoy索引
        :param load_model:
        :return:
        """
        ent_weight = load_model.entityEmbedding.weight.detach().cpu().numpy()
        annoy_index = AnnoyIndex(100)
        for idx, (ent, embed) in enumerate(dict(zip(self.entity_dict.keys(), ent_weight)).items()):
            annoy_index.add_item(idx, embed)
        annoy_index.build(200)
        if not os.path.exists(self.config.annoy_index_path):
            os.makedirs(self.config.annoy_index_path)
        
        #joblib.dump(annoy_index,os.path.join(self.config.annoy_index_path, "annoy_index.pkl"))
        annoy_index.save(os.path.join(self.config.annoy_index_path, "annoy_index.index"))

    def similarity_topk(self, key, top_k):
        """
        传统方式,找到与key相似的节点
        :param key:
        :param top_k:
        :return:
        """
        entity_embed_dict = dict()
        similarity_topk = dict()

        with open(os.path.join(self.config.dump_embedding_path, "entity_embedding.txt"), "r",
                  encoding="utf-8") as f:
            for line in f.readlines():
                json_dict = json.loads(line.strip(), encoding="utf-8")
                entity_embed_dict[json_dict["word"]] = np.array(json_dict["word_vet"])

        key_embedding = entity_embed_dict.get(key)
        for k, v in entity_embed_dict.items():
            if k == key:
                similarity_topk[k] = 0
            else:
                distance = calculate_distance(key_embedding, v)
                similarity_topk[k] = distance
        similarity_topk = sorted(similarity_topk.items(), key=lambda x: x[1], reverse=True)
        return {i[0]: i[1] for i in similarity_topk[:top_k]}

    def similarity_topk_annoy(self, key, top_k):
        """
        annoy方式,找到与key相似的节点
        :param key:
        :param top_k:
        :return:
        """
        annoy_index = AnnoyIndex(100)
        annoy_index.load(os.path.join(self.config.annoy_index_path, "annoy_index.index"))
        key_value = self.entity_dict.get(key, None)
        if key_value:
            res = annoy_index.get_nns_by_item(self.entity_dict[key], top_k)
            for i in res:
                print(self.id2entity[i])


if __name__ == '__main__':
    tr = TransX()
    tr.train()
