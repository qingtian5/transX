# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: KG2E.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class KG2E(nn.Module):
    def __init__(self, entity_num, relation_num, embedding_dim, margin=1.0, sim="KL", vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        assert (sim in ["KL", "EL"])
        self.margin = margin
        self.sim = sim
        self.ke = embedding_dim
        self.vmin = vmin
        self.vmax = vmax

        # nn.Embedding具有一个权重, 形状是(num_embeddings, embedding_dim);
        # Embedding的输入形状是N*W, N是bach_size, W是序列长度, 输出的形状是: N*W*embedding_dim
        self.entityEmbedding = nn.Embedding(num_embeddings=entity_num, embedding_dim=embedding_dim)
        self.entityCovar = nn.Embedding(num_embeddings=entity_num, embedding_dim=embedding_dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=embedding_dim)
        self.relationCovar = nn.Embedding(num_embeddings=relation_num, embedding_dim=embedding_dim)

    def forward(self, pos_x, neg_x):
        """
       前向传播
       :param pos_x: (torch.tensor)The positive triples tensor, shape(batch_size, 3)
       :param neg_x: (torch.tensor)The negtive triples tensor, shape(batch_size, 3)
       :return:
       """
        size = pos_x.size()[0]

        # Calculate score
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        return torch.sum(F.relu(input=pos_score - neg_score + self.margin)) / size

    def KLScore(self, **kwargs):
        """
        Calculate the KL loss between T-H distribution and R distribution.
        There are four parts in loss function.
        :param kwargs:
        :return:
        """
        # Calculate KL(e, r)
        losep1 = torch.sum(kwargs["errorv"] / kwargs["relationv"], dim=1)
        losep2 = torch.sum((kwargs["relationm"] - kwargs["errorm"]) ** 2 / kwargs["relationv"], dim=1)
        KLer = (losep1 + losep2 - self.ke) / 2

        # Calculate KL(r, e)
        losep1 = torch.sum(kwargs["relationv"] / kwargs["errorv"], dim=1)
        losep2 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / kwargs["errorv"], dim=1)
        KLre = (losep1 + losep2 - self.ke) / 2
        return (KLer + KLre) / 2

    def ELScore(self, **kwargs):
        """
        Calculate the EL loss between T-H distribution and R distribution.
        There are three parts in loss function.
        :param kwargs:
        :return:
        """
        losep1 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / (kwargs["errorv"] + kwargs["relationv"]),
                           dim=1)
        losep2 = torch.sum(torch.log(kwargs["errorv"] + kwargs["relationv"]), dim=1)
        return (losep1 + losep2) / 2

    def score_op(self, input_triples):
        """
        Calculate the score of triples
            Step1: Split input as head, relation and tail index
            Step2: Transform index tensor to embedding
            Step3: Calculate the score with "KL" or "EL"
            Step4: Return the score
        :param input_triples:
        :return:
        """
        head, relation, tail = torch.chunk(input=input_triples, chunks=3, dim=1)

        headm = torch.squeeze(self.entityEmbedding(head), dim=1)
        headv = torch.squeeze(self.entityCovar(head), dim=1)
        tailm = torch.squeeze(self.entityEmbedding(tail), dim=1)
        tailv = torch.squeeze(self.entityCovar(tail), dim=1)
        relationm = torch.squeeze(self.relationEmbedding(relation), dim=1)
        relationv = torch.squeeze(self.relationCovar(relation), dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        if self.sim == "KL":
            return self.KLScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        elif self.sim == "EL":
            return self.ELScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        else:
            print("ERROR : Sim %s is not supported!" % self.sim)
            exit(1)

    def normalize_embedding(self):
        self.entityEmbedding.weight.data.copy_(
            torch.renorm(input=self.entityEmbedding.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))

        self.relationEmbedding.weight.data.copy_(
            torch.renorm(input=self.relationEmbedding.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))

        self.entityCovar.weight.data.copy_(
            torch.clamp(input=self.entityCovar.weight.detach().cpu(), min=self.vmin, max=self.vmax))

        self.relationCovar.weight.data.copy_(
            torch.clamp(input=self.relationCovar.weight.detach().cpu(), min=self.vmin, max=self.vmax))

    def ret_eval_weights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "entityCovar": self.entityCovar.weight.detach().cpu().numpy(),
                "relationCovar": self.relationCovar.weight.detach().cpu().numpy(),
                "Sim": self.sim}
