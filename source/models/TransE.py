# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: TransE.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, entity_num, relation_num, embedding_dim, margin=1.0):
        super(TransE, self).__init__()
        self.margin = margin

        # nn.Embedding具有一个权重, 形状是(num_embeddings, embedding_dim);
        # Embedding的输入形状是N*W, N是bach_size, W是序列长度, 输出的形状是: N*W*embedding_dim
        self.entityEmbedding = nn.Embedding(num_embeddings=entity_num, embedding_dim=embedding_dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=embedding_dim)

        # 使用p-2范数计算向量x和y之间的批内成对距离
        self.dist_fn = nn.PairwiseDistance(2)

    def forward(self, pos_x, neg_x):
        """
        前向传播
        :param pos_x: 正样本, shape(batch_size, 3)
        :param neg_x: 负样本, shape(batch_size, 3)
        :return:
        """
        size = pos_x.size()[0]

        # 计算正负样本"头实体+关系-->>尾实体"的L2范数(距离)
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        #  损失函数: Hinge Loss, 通常被用于最大间隔算法 max margin
        #  max(posScore-negScore+margin, 0)
        return torch.sum(F.relu(input=pos_score - neg_score + self.margin)) / size

    def score_op(self, input_triple):
        """
        计算正负样本"头实体+关系-->>尾实体"的L2范数(距离)
        :param input_triple:
        :return:
        """

        # Step1. 将tensor按dim（行或列）分割成chunks个tensor块，返回的是一个元组
        # head/relation/tail: shape(batch_size, 1)
        head, relation, tail = torch.chunk(input=input_triple, chunks=3, dim=1)

        # Step2. 对数据的维度进行压缩，去掉维数为1的的维度
        # head/relation/tail: shape(batch_size, 1) --> shape(batch_size, 1, embedDim) --> shape(batch_size, embedDim)
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)

        # Step3. 计算距离
        # output:  shape(batch_size, embedDim) --> shape(batch_size, 1)
        output = self.dist_fn(head + relation, tail)
        return output

    def normalize_embedding(self):
        """
        In every training epoch, the entity embedding should be normalize
             Step1: Get numpy.array from embedding weight
             Step2: Normalize array
             Step3: Assign normalized array to embedding

        :return:
        """
        embed_weight = self.entityEmbedding.weight.detach().cpu().numpy()
        embed_weight = embed_weight / np.sqrt(np.sum(np.square(embed_weight), axis=1, keepdims=True))
        self.entityEmbedding.weight.data.copy_(torch.from_numpy(embed_weight))

    def ret_eval_weights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy()}
