# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/12/8 21:09
# @file: TransD.py

'''TransD是知识图谱表示学习中的一种模型，它通过将实体和关系投影到不同的空间中来捕捉实体和关系之间的语义信息'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation import cal_map_sim_tensor as cal_map_sim
from evaluation import cal_rank_tensor as cal_rank
from tqdm import trange, tqdm
import time


class TransD(nn.Module):
    def __init__(self, entity_num, relation_num, entity_dim, relation_dim, margin=1.0): #margin用于控制正负样本之间的间隔的边界，默认值为1.0
        super(TransD, self).__init__() #这是调用父类（nn.Module）的构造函数，确保正确地初始化了TransD类
        self.margin = margin

        # nn.Embedding具有一个权重, 形状是(num_embeddings, embedding_dim);
        # Embedding的输入形状是N*W, N是bach_size, W是序列长度, 输出的形状是: N*W*embedding_dim
        self.entityEmbedding = nn.Embedding(num_embeddings=entity_num, embedding_dim=entity_dim)
        self.entityMapEmbedding = nn.Embedding(num_embeddings=entity_num, embedding_dim=entity_dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=relation_dim)
        self.relationMapEmbedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=relation_dim)

        # 使用p-2范数计算向量x和y之间的批内成对距离,它计算两个输入张量之间的距离
        #调用方法计算头实体、关系与尾实体之间的距离，以评估正负样本之间的差异。
        self.dist_fn = nn.PairwiseDistance(2)

    def score_op(self, input_triples):
        """
        计算正负样本"头实体+关系-->>尾实体"的L2范数(距离)
        :param input_triples:
        :return:
        """

        # Step1. 将tensor按dim（行或列）分割成chunks个tensor块，返回的是一个元组
        # head/relation/tail: shape(batch_size, 1)
        head, relation, tail = torch.chunk(input_triples, chunks=3, dim=1)

        # Step2. 对数据的维度进行压缩，去掉维数为1的的维度
        #实体索引转换为实体嵌入的过程： shape(batch_size, 1) --> shape(batch_size, 1, embedDim) --> shape(batch_size, embedDim)
        head_p = torch.squeeze(self.entityMapEmbedding(head), dim=1) #self.entityMapEmbedding(head)这一步将每个实体索引映射为一个嵌入向量
        head = torch.squeeze(self.entityEmbedding(head), dim=1)

        tail_p = torch.squeeze(self.entityMapEmbedding(tail), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)

        relation_p = torch.squeeze(self.relationMapEmbedding(relation), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)

        # head_p:    shape(batch_size, embedDim) --> shape(batch_size, 1, embedDim)
        # tail_p:    shape(batch_size, embedDim) --> shape(batch_size, 1, embedDim)
        # relation_p:shape(batch_size, embedDim) --> shape(batch_size, embedDim, 1)
        # shape(batch_size, embedDim, 1) * shape(batch_size, 1, embedDim) ->  shape(batch_size, embedDim, embedDim)
        #上述乘积表示两个嵌入之间的内积或外积关系，例如在TransD模型中，用于计算头尾实体之间的变换关系
        head_p = torch.unsqueeze(head_p, dim=1)
        tail_p = torch.unsqueeze(tail_p, dim=1)
        relation_p = torch.unsqueeze(relation_p, dim=2)

        relation_dim = relation.size()[1] #计算了关系嵌入和实体嵌入的维度
        entity_dim = head.size()[1]
        if input_triples.is_cuda:
            '''计算关系嵌入和头尾实体嵌入的变换矩阵
            torch.eye(relation_dim, entity_dim) 创建了一个单位矩阵，形状为 (relation_dim, entity_dim)。这个单位矩阵用于保证矩阵乘法的稳定性。
            '''
            Mrh = torch.matmul(relation_p, head_p) + torch.eye(relation_dim, entity_dim) \
                .cuda(input_triples.device.index)
            Mrt = torch.matmul(relation_p, tail_p) + torch.eye(relation_dim, entity_dim) \
                .cuda(input_triples.device.index)
        else:
            Mrh = torch.matmul(relation_p, head_p) + torch.eye(relation_dim, entity_dim)
            Mrt = torch.matmul(relation_p, tail_p) + torch.eye(relation_dim, entity_dim)

        # Mrh/Mrt : shape(batch_size, embedDim, embedDim)
        # head/tail : shape(batch_size, embedDim) --> shape(batch_size, embedDim, 1)
        # (B, Em, En) * (B, En, 1) --> (B, Em, 1) --> (B, Em)
        '''变换矩阵与头尾实体嵌入的矩阵乘法'''
        head = torch.unsqueeze(head, dim=2)
        tail = torch.unsqueeze(tail, dim=2)
        head = torch.squeeze(torch.matmul(Mrh, head), dim=2) #(B, Em, 1) --> (B, Em)
        tail = torch.squeeze(torch.matmul(Mrt, tail), dim=2)

        output = self.dist_fn(head + relation, tail) #head + relation 表示头实体嵌入与关系嵌入之间的求和操作
        return output

    def normalize_embedding(self):
        """
        在每个训练周期中，实体嵌入应该被标准化
            步骤1：从嵌入权重中获取 numpy.array
            步骤2：标准化数组
            步骤3：将标准化后的数组赋值给嵌入
        :return:
        """
        # 将实体嵌入张量中的数据拷贝到CPU，并标准化为L2范数为1的向量
        '''torch.renorm 函数用于对张量进行重新标准化
        首先将实体嵌入权重从GPU上取回到CPU，并调用 detach() 方法来确保不会在此过程中计算梯度。这一步是因为 torch.renorm 函数只能应用于不需要梯度的张量
        p=2：使用 L2 范数来度量向量的大小
        dim=0：标准化操作将应用在张量的第一个维度上，即每个嵌入向量的维度上
        maxnorm=1：每个嵌入向量的L2范数将被调整为1，以确保它们具有相同的比例
        最后用copy_ 方法用于将标准化后的嵌入矩阵拷贝回实体嵌入的权重中，以更新模型中的参数'''
        self.entityEmbedding.weight.data.copy_(
            torch.renorm(input=self.entityEmbedding.weight.detach().cpu(), p=2, dim=0, maxnorm=1))

        # 将关系嵌入张量中的数据拷贝到CPU，并标准化为L2范数为1的向量
        self.relationEmbedding.weight.data.copy_(
            torch.renorm(input=self.relationEmbedding.weight.detach().cpu(), p=2, dim=0, maxnorm=1))

    def ret_eval_weights(self):
        """
        Return evaluation weights
        :return:
        """
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "entityMapEmbed": self.entityMapEmbedding.weight.detach().cpu().numpy(),
                "relationMapEmbed": self.relationMapEmbedding.weight.detach().cpu().numpy()}

    def forward(self, pos_x, neg_x):
        """
       前向传播
       :param pos_x: (torch.tensor)正样本三元组张量, shape(batch_size, 3)
       :param neg_x: (torch.tensor)负样本三元组张量, shape(batch_size, 3)
       :return:
       """
        ## 获取正样本的批量大小
        size = pos_x.size()[0]

        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        ## 计算损失函数，包括margin损失和激活函数ReLU
        return torch.sum(F.relu(input=pos_score - neg_score + self.margin)) / size #对这些值进行求和，并除以批量大小 size 来计算平均损失

    def compute_mr(self, data_loader, device, config):
        """ 占用 60G 显存 """
        self.eval()
        r, n = 0, 0
        # start = time.time()
        # length = len(data_loader)
        for idx, tri in enumerate(tqdm(data_loader, desc="valid_mr_score")):
            tri = torch.tensor(tri, dtype=torch.long, device=device)
            head, relation, tail = tri[:, 0], tri[:, 1], tri[:, 2]
            headp = torch.index_select(self.entityMapEmbedding.weight, 0, head)
            head = torch.index_select(self.entityEmbedding.weight, 0, head)
            relationp = torch.index_select(self.relationMapEmbedding.weight, 0, relation)
            relation = torch.index_select(self.relationEmbedding.weight, 0, relation)

            reldim = relation.shape[1]
            entdim = head.shape[1]
            Mrh = torch.matmul(relationp[:, :, None], headp[:, None, :]) + torch.eye(reldim, entdim, device=relation.device)
            head = torch.squeeze(torch.matmul(Mrh, head[:, :, None]), 2)
            simScore = cal_map_sim(head + relation, self.entityEmbedding.weight, self.entityMapEmbedding.weight, relationp, simMeasure=config.sim_measure)
            ranks = cal_rank(simScore, tail, simMeasure=config.sim_measure)
            r += torch.sum(ranks).item()
            n += ranks.shape[0]
            # if idx % 100 == 0:
            #     print(f"Iter: {idx} | {length}, cost time {time.time() - start} per 100 iters")
            #     start = time.time()
        print(f"MR: {r / n}")
        return r / n
    