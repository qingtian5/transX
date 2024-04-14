# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import dataloader
from tqdm import tqdm
import torch


'''mr_evaluation: 该函数是评估函数的入口，接受数据加载器、模型名称以及相似度度量方法作为输入，对整个数据集进行评估'''
def mr_evaluation(eval_loader: dataloader, model, sim_measure="dot", **kwargs):
    if model not in ["TransE", "TransH", "TransD", "TransA", "KG2E"]:
        ValueError("ERROR : The %s evaluation is not supported!" % model)

    r, n = 0, 0
    for tri in eval_loader:
        tri = tri.numpy()
        head, relation, tail = tri[:, 0], tri[:, 1], tri[:, 2]
        if model == "TransE":
            ranks = eval_transe(head, relation, tail, sim_measure, **kwargs)
        elif model == "TransH":
            ranks = eval_transh(head, relation, tail, sim_measure, **kwargs)
        elif model == "TransD":
            ranks = eval_transd(head, relation, tail, sim_measure, **kwargs)
        elif model == "TransA":
            ranks = eval_transa(head, relation, tail, **kwargs)
        else:
            ranks = eval_kg2e(head, relation, tail, **kwargs)

        r += np.sum(ranks)
        n += ranks.shape[0]
    return r / n


def eval_transe(head, relation, tail, sim_measure, **kwargs):
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Calculate the similarity, sort the score and get the rank
    sim_score = cal_similarity(head + relation, kwargs["entityEmbed"], simMeasure=sim_measure)
    ranks = cal_rank(sim_score, tail, simMeasure=sim_measure)
    return ranks

#计算两个向量集之间的相似性得分
def cal_similarity(expTailMatrix: np.ndarray, tailEmbedding: np.ndarray, simMeasure="dot"):
    if simMeasure == "dot":
        return np.matmul(expTailMatrix, tailEmbedding.T)
    elif simMeasure == "cos":
        # First, normalize expTailMatrix and tailEmbedding
        # Then, use dot to calculate similarity
        return np.matmul(expTailMatrix / np.linalg.norm(expTailMatrix, ord=2, axis=1, keepdims=True),
                         (tailEmbedding / np.linalg.norm(expTailMatrix, ord=2, axis=1, keepdims=True)).T)
    elif simMeasure == "L2":
        sim_score = []
        for expM in expTailMatrix:
            score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=2, axis=1, keepdims=False)
            sim_score.append(score)
        return np.array(sim_score)
    elif simMeasure == "L1":
        sim_score = []
        for expM in expTailMatrix:
            score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=1, axis=1, keepdims=False)
            sim_score.append(score)
        return np.array(sim_score)
    else:
        print("ERROR : Similarity method %s is not supported!" % simMeasure)
        exit(1)

#根据相似性得分和给定的尾实体，计算每个尾实体的排名
def cal_rank(simScore: np.ndarray, tail: np.ndarray, simMeasure: str):
    realScore = simScore[np.arange(tail.shape[0]), tail].reshape((-1, 1))
    #函数计算每个尾实体与目标尾实体的得分差异矩阵judMatrix
    judMatrix = simScore - realScore
    if simMeasure == "dot" or simMeasure == "cos":
        '''
        The larger the score, the better the rank.
        '''
        judMatrix[judMatrix > 0] = 1
        judMatrix[judMatrix < 0] = 0
        judMatrix = np.sum(judMatrix, axis=1)
        return judMatrix
    elif simMeasure == "L2" or simMeasure == "L1":
        '''
        The smaller the score, the better the rank
        '''
        judMatrix[judMatrix > 0] = 0
        judMatrix[judMatrix < 0] = 1
        judMatrix = np.sum(judMatrix, axis=1)
        return judMatrix
    else:
        print("ERROR : Similarity measure is not supported!")
        exit(1)

#计算超平面投影后的相似性得分
def cal_hyper_sim(expTailMatrix, tailEmbedding, hyperMatrix, simMeasure="dot"):
    simScore = []
    for expM, hypM in zip(expTailMatrix, hyperMatrix):
        '''
        expM : shape(E,)
        hypM : shape(E,)
        Step1 : Projection tailEmbedding on hyperM as hyperTailEmbedding(shape(N,E))
        Step2 : Calculate similarity between expTailMatrix and hyperTailEmbedding
        Step3 : Add similarity to simMeasure
        (1, E) * matmul((N, E), (E, 1)) -> (1, E) * (N, 1) -> (N, E)
        Step1：将tailEmbedding投影到超平面hypM上，得到超平面尾实体嵌入向量hyperEmbedding
        Step2：计算expTailMatrix与hyperEmbedding之间的相似性得分。
        Step3：根据指定的相似性度量方法，将相似性得分添加到simScore列表中
        '''
        hyperEmbedding = tailEmbedding - hypM[np.newaxis, :] * np.matmul(tailEmbedding, hypM[:, np.newaxis])
        if simMeasure == "dot":
            simScore.append(np.squeeze(np.matmul(hyperEmbedding, expM[:, np.newaxis])))
        elif simMeasure == "L2":
            # (E,) -> (1, E)
            # (N, E) - (1, E) -> (N, E)
            # np.linalg.norm()
            score = np.linalg.norm(hyperEmbedding - expM[np.newaxis, :], ord=2, axis=1, keepdims=False)
            simScore.append(score)
        else:
            print("ERROR : simMeasure %s is not supported!" % simMeasure)
            exit(1)
    return np.array(simScore)


'''
Evaluation for TransH
'''


def eval_transh(head, relation, tail, simMeasure, **kwargs):
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    hyper = np.take(kwargs["hyperEmbed"], indices=relation, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Projection
    head = head - hyper * np.sum(hyper * head, axis=1, keepdims=True)
    simScore = cal_hyper_sim(head + relation, kwargs["entityEmbed"], hyper, simMeasure=simMeasure)
    ranks = cal_rank(simScore, tail, simMeasure=simMeasure)
    return ranks


def cal_map_sim(expTailMatrix, tailEmbedding, tailMapMatrix, relMapMatrix, simMeasure="L2"):
    simScore = []
    for expM, relMap in zip(expTailMatrix, relMapMatrix):
        # relMap : (Em, ) -> (1, Em, 1)
        # tailMapMatrix : (N, En) -> (N, 1, En)
        # (1, Em, 1) * (N, 1, En) -> (N, Em, En)
        # (N, Em, En) * (N, En, 1) -> (N, Em, 1)
        # expM : (Em, )
        entdim = tailEmbedding.shape[1]
        reldim = relMapMatrix.shape[1]
        Mrt = np.matmul(relMap[np.newaxis, :, np.newaxis], tailMapMatrix[:, np.newaxis, :]) + np.eye(reldim, entdim)
        if simMeasure == "L2":
            score = np.linalg.norm(np.squeeze(np.matmul(Mrt, tailEmbedding[:, :, np.newaxis]), axis=2) - expM, ord=2,
                                   axis=1, keepdims=False)
            simScore.append(score)
        elif simMeasure == "L1":
            score = np.linalg.norm(np.squeeze(np.matmul(Mrt, tailEmbedding[:, :, np.newaxis]), axis=2) - expM, ord=1,
                                   axis=1, keepdims=False)
            simScore.append(score)
        else:
            print("ERROR : SimMeasure %s is not supported!" % simMeasure)
            exit(1)
    return np.array(simScore)


# cal_map_sim 转化为使用 tensor 操作的模式
def cal_map_sim_tensor(expTailMatrix, tailEmbedding, tailMapMatrix, relMapMatrix, simMeasure="L2"):
    simScore = []
    for expM, relMap in zip(expTailMatrix, relMapMatrix):
        # relMap : (Em, ) -> (1, Em, 1)
        # tailMapMatrix : (N, En) -> (N, 1, En)
        # (1, Em, 1) * (N, 1, En) -> (N, Em, En)
        # (N, Em, En) * (N, En, 1) -> (N, Em, 1)
        # expM : (Em, )
        entdim = tailEmbedding.shape[1]
        reldim = relMap.shape[0]
        relMap = relMap.unsqueeze(0).unsqueeze(2)  # (1, Em, 1)
        tailMapMatrix_unsqueezed = tailMapMatrix.unsqueeze(1)  # (N, 1, En)

        Mrt = torch.matmul(relMap, tailMapMatrix_unsqueezed) + torch.eye(reldim, entdim, device=tailEmbedding.device)

        tailEmbedding_unsqueezed = tailEmbedding.unsqueeze(2)  # (N, En, 1)
        transformed_tail = torch.squeeze(torch.matmul(Mrt, tailEmbedding_unsqueezed), 2)  # (N, Em)
        expM = expM.unsqueeze(0).expand_as(transformed_tail)  # Expand expM to match the dimensions

        if simMeasure == "L2":
            score = torch.norm(transformed_tail - expM, p=2, dim=1)
            simScore.append(score)
        elif simMeasure == "L1":
            score = torch.norm(transformed_tail - expM, p=1, dim=1)
            simScore.append(score)
        else:
            print(f"ERROR: SimMeasure {simMeasure} is not supported!")
            exit(1)
    return torch.stack(simScore)


# cal_rank 转化为 tensor 操作
def cal_rank_tensor(simScore: torch.Tensor, tail: torch.Tensor, simMeasure: str):
    # Real score from simScore at positions specified by tail
    realScore = simScore[torch.arange(tail.shape[0]), tail].unsqueeze(1)

    # Compute the matrix of score differences
    judMatrix = simScore - realScore

    if simMeasure in ["dot", "cos"]:
        '''
        The larger the score, the better the rank.
        Convert differences to binary rank contributions.
        '''
        judMatrix = (judMatrix > 0).float()  # Convert condition to float for sum
        ranks = judMatrix.sum(dim=1)
        return ranks
    elif simMeasure in ["L2", "L1"]:
        '''
        The smaller the score, the better the rank
        Convert differences to binary rank contributions.
        '''
        judMatrix = (judMatrix < 0).float()  # Convert condition to float for sum
        ranks = judMatrix.sum(dim=1)
        return ranks
    else:
        print("ERROR: Similarity measure is not supported!")
        exit(1)

#transD模型评估
def eval_transd(head, relation, tail, simMeasure, **kwargs):
    # Gather embedding
    headp = np.take(kwargs["entityMapEmbed"], indices=head, axis=0)
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relationp = np.take(kwargs["relationMapEmbed"], indices=relation, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Generate Mapping Matrix
    # (B, Em) -> (B, Em, 1), (B, En) -> (B, 1, En)
    # (B, Em, 1) * (B, 1, En) -> (B, Em, En)
    # (B, En) -> (B, En, 1)
    # (B, Em, En) * (B, En, 1) -> (B, Em, 1)
    reldim = relation.shape[1]
    entdim = head.shape[1]
    #映射矩阵Mrh
    Mrh = np.matmul(relationp[:, :, np.newaxis], headp[:, np.newaxis, :]) + np.eye(reldim, entdim) #np.newaxis用于为数组添加一个新的轴
    head = np.squeeze(np.matmul(Mrh, head[:, :, np.newaxis]), axis=2)
    simScore = cal_map_sim(head + relation, kwargs["entityEmbed"], kwargs["entityMapEmbed"], relationp,
                           simMeasure=simMeasure)
    ranks = cal_rank(simScore, tail, simMeasure=simMeasure)
    return ranks


def cal_weight_sim(expTailMatrix, tailEmbedding, Wr):
    simScore = []
    for expM in tqdm(zip(expTailMatrix)):
        # expM : (E, )
        # (N, E) - (E, ) -> (N, E) -> abs() -> (N, E, 1), (N, 1, E)
        # (N, 1, E) * (N, E, E) * (N, E, 1) -> (N, 1, 1)
        error = np.abs(tailEmbedding - expM)
        score = np.squeeze(np.matmul(np.matmul(error[:, np.newaxis, :], Wr), error[:, :, np.newaxis]))
        simScore.append(score)
    return np.array(simScore)


def eval_transa(head, relation, tail, **kwargs):
    # Gather embedding
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Calculate simScore
    simScore = cal_weight_sim(head + relation, kwargs["entityEmbed"], kwargs["Wr"])
    ranks = cal_rank(simScore, tail, simMeasure="L2")
    return ranks


def cal_kl_sim(headMatrix, headCoMatrix, relationMatrix, relationCoMatrix, tailMatrix, tailCoMatrix, simMeasure="KL"):
    simScore = []
    for hM, hC, rM, rC in tqdm(zip(headMatrix, headCoMatrix, relationMatrix, relationCoMatrix)):
        # (N, E) - (E, )
        # (N, E) + (E, )
        errorm = tailMatrix - hM
        errorv = tailCoMatrix + hC
        if simMeasure == "KL":
            # (N, E) / (E, ) -> (N, E) -> sum() -> (N, )
            # (N, E) - (E, ) -> (N, E) ** 2 / (E, )
            score1 = np.sum(errorv / rC, axis=1, keepdims=False) + \
                     np.sum((rM - errorm) ** 2 / rC, axis=1, keepdims=False)
            score2 = np.sum(rC / errorv, axis=1, keepdims=False) + \
                     np.sum((rM - errorm) ** 2 / errorv, axis=1, keepdims=False)
            simScore.append((score1 + score2) / 2)
        elif simMeasure == "EL":
            score1 = np.sum((errorm - rM) ** 2 / (errorv + rC), axis=1, keepdims=False)
            score2 = np.sum(np.log(errorv + rC), axis=1, keepdims=False)
            simScore.append((score1 + score2) / 2)
    return np.array(simScore)


def eval_kg2e(head, relation, tail, **kwargs):
    # Gather embedding
    headv = np.take(kwargs["entityCovar"], indices=head, axis=0)
    headm = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relationv = np.take(kwargs["relationCovar"], indices=relation, axis=0)
    relationm = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Calculate simScore
    simScore = cal_kl_sim(headm, headv, relationm, relationv, kwargs["entityEmbed"], kwargs["entityCovar"],
                          simMeasure=kwargs["Sim"])
    ranks = cal_rank(simScore, tail, simMeasure="L2")
    return ranks
