# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: TransH.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransH(nn.Module):
    def __init__(self, entity_num, relation_num, embedding_dim, margin=1.0, C=1.0, eps=0.001):
        super(TransH, self).__init__()
        self.ent_num = entity_num
        self.rel_num = relation_num
        self.margin = margin
        self.C = C
        self.eps = eps

        # nn.Embedding具有一个权重, 形状是(num_embeddings, embedding_dim);
        # Embedding的输入形状是N*W, N是bach_size, W是序列长度, 输出的形状是: N*W*embedding_dim
        self.entityEmbedding = nn.Embedding(num_embeddings=entity_num, embedding_dim=embedding_dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=embedding_dim)
        self.relationHyper = nn.Embedding(num_embeddings=relation_num, embedding_dim=embedding_dim)

        # 使用p-2范数计算向量x和y之间的批内成对距离
        self.dist_fn = nn.PairwiseDistance(2)

    def forward(self, pos_x, neg_x):
        """
        前向传播
        :param pos_x: (torch.tensor)The positive triples tensor, shape(batch_size, 3)
        :param neg_x: (torch.tensor)The negative triples tensor, shape(batch_size, 3)
        :return:
        """
        size = pos_x.size()[0]

        # 计算正负样本"头实体+关系-->>尾实体"的L-?范数(距离),头尾实体是在关系超平面上的映射
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        margin_loss = torch.sum(F.relu(input=pos_score - neg_score + self.margin))
        entity_loss = torch.sum(F.relu(torch.norm(self.entityEmbedding.weight, p=2, dim=1, keepdim=False) - 1))
        loss = torch.sum(
            F.relu(torch.sum(self.relationHyper.weight * self.relationEmbedding.weight, dim=1, keepdim=False) / \
                   torch.norm(self.relationEmbedding.weight, p=2, dim=1, keepdim=False) - self.eps ** 2))
        return margin_loss / size + self.C * (entity_loss / self.ent_num + loss / self.rel_num)

    def score_op(self, input_triple):
        """
        计算正负样本"头实体+关系-->>尾实体"的L2范数(距离)
        :param input_triple:
        :return:
        """

        # Step1. 将tensor按dim（行或列）分割成chunks个tensor块，返回的是一个元组
        # head/relation/tail: shape(batch_size, 1)
        head, relation, tail = torch.chunk(input_triple, chunks=3, dim=1)

        # Step2. 对数据的维度进行压缩，去掉维数为1的的维度
        # head/relation/tail: shape(batch_size, 1) --> shape(batch_size, 1, embedDim) --> shape(batch_size, embedDim)
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        rel_hyper = torch.squeeze(self.relationHyper(relation), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)

        # Step3. 将头实体和尾实体嵌入到关系超平面
        head = head - rel_hyper * torch.sum(head * rel_hyper, dim=1, keepdim=True)
        tail = tail - rel_hyper * torch.sum(tail * rel_hyper, dim=1, keepdim=True)

        # Step4. 计算距离
        return self.dist_fn(head + relation, tail)

    def normalize_embedding(self):
        """
        In every training epoch, the entity embedding should be normalize
             Step1: Get numpy.array from embedding weight
             Step2: Normalize array
             Step3: Assign normalized array to embedding

        :return:
        """
        hyper_weight = self.relationHyper.weight.detach().cpu().numpy()
        hyper_weight = hyper_weight / np.sqrt(np.sum(np.square(hyper_weight), axis=1, keepdims=True))
        self.relationHyper.weight.data.copy_(torch.from_numpy(hyper_weight))

    def ret_eval_weights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "hyperEmbed": self.relationHyper.weight.detach().cpu().numpy()}
