# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: TransA.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransA(nn.Module):
    def __init__(self, entity_num, relation_num, embedding_dim, margin=1.0, lamb=0.01, C=0.2):
        super(TransA, self).__init__()
        self.ent_num = entity_num
        self.rel_num = relation_num
        self.emb_dim = embedding_dim
        self.margin = margin
        self.lamb = lamb
        self.C = C

        # nn.Embedding具有一个权重, 形状是(num_embeddings, embedding_dim);
        # Embedding的输入形状是N*W, N是bach_size, W是序列长度, 输出的形状是: N*W*embedding_dim
        self.entityEmbedding = nn.Embedding(num_embeddings=entity_num, embedding_dim=embedding_dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=embedding_dim)

    def forward(self, pos_x, neg_x):
        """
        前向传播
        :param pos_x: (torch.tensor)The positive triples tensor, shape(batch_size, 3)
        :param neg_x: (torch.tensor)The negtive triples tensor, shape(batch_size, 3)
        :return:
        """
        size = pos_x.size()[0]
        self.calculate_wr(pos_x, neg_x)

        # Calculate score
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        # Calculate loss
        margin_loss = 1 / size * torch.sum(F.relu(input=pos_score - neg_score + self.margin))
        wr_loss = 1 / size * torch.norm(input=self.Wr, p=self.L)
        weight_loss = (1 / self.ent_num * torch.norm(input=self.entityEmbedding.weight, p=2) + \
                       1 / self.rel_num * torch.norm(input=self.relationEmbedding.weight, p=2))
        return margin_loss + self.lamb * wr_loss + self.C * weight_loss

    def normalize_embedding(self):
        pass

    def ret_eval_weights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "Wr": self.Wr.detach().cpu().numpy()}

    def calculate_wr(self, pos_x, neg_x):
        """
        Calculate the Mahalanobis distance weights
        :param pos_x:
        :param neg_x:
        :return:
        """
        pos_head, pos_rel, pos_tail = torch.chunk(input=pos_x, chunks=3, dim=1)
        neg_head, neg_rel, neg_tail = torch.chunk(input=neg_x, chunks=3, dim=1)

        pos_head_m, pos_rel_m, pos_tail_m = self.entityEmbedding(pos_head), self.relationEmbedding(pos_rel), \
                                            self.entityEmbedding(pos_tail)
        neg_head_m, neg_rel_m, neg_tail_m = self.entityEmbedding(neg_head), self.relationEmbedding(neg_rel), \
                                            self.entityEmbedding(neg_tail)

        error_pos = torch.abs(pos_head_m + pos_rel_m - pos_tail_m)
        error_neg = torch.abs(neg_head_m + neg_rel_m - neg_tail_m)
        del pos_head_m, pos_rel_m, pos_tail_m, neg_head_m, neg_rel_m, neg_tail_m

        self.Wr[pos_rel] += torch.sum(torch.matmul(error_neg.permute((0, 2, 1)), error_neg), dim=0) - \
                            torch.sum(torch.matmul(error_pos.permute((0, 2, 1)), error_pos), dim=0)

    def score_op(self, input_triples):
        """

        :param input_triples:
        :return:
        """
        head, relation, tail = torch.chunk(input=input_triples, chunks=3, dim=1)
        rel_wr = self.Wr[relation]
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)

        # (B, E) -> (B, 1, E) * (B, E, E) * (B, E, 1) -> (B, 1, 1) -> (B, )
        error = torch.unsqueeze(torch.abs(head + relation - tail), dim=1)
        error = torch.matmul(torch.matmul(error, torch.unsqueeze(rel_wr, dim=0)), error.permute((0, 2, 1)))
        return torch.squeeze(error)
