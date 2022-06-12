# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.02.20
@notice  : HoGCN (Homogeneous Graph Convolutional Networks)
"""


import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from .BasicModule import BasicModule
import numpy as np
from config import opt


def calAdjacencyMatrix(adjacency, node, edgeConnectMode):
    """
    calculate the weighted adjacency matrix using the attention machanism
    adjacency: adjacency matrix, shape: [B, maxLength, maxLength] (values are 0/1)
    node: node feature matrix, shape: [B, maxLength, C]
    edgeConnectMode: edge connection mode, "Geo" or "Full"(non-local connection)

    if nodes v_1 and v_2 are connected; feature vectors are x, y;
    the calculation of the edge weight e_12 between v_1 and v_2:
    e_12 = exp(x^T·y)
    """

    # non-local connection
    if edgeConnectMode == "Full":
        adjacency = torch.ones_like(adjacency).to(adjacency.device)

    # normalize node features
    node = F.normalize(node, p=2, dim=2)

    # e_12 = exp(x^T·y)
    A = torch.bmm(node, node.permute(0, 2, 1))
    A = torch.exp(A)
    A = torch.mul(A, adjacency)
    return A


def calNormalizedA(A):
    """
    calculate the normalized adjacency matrix
    A: adjacency matrix, shape: [B, maxLength, maxLength] (values are 0/1)
    """
    # 1. ATilde = A + I
    ATilde = A.to(torch.float32)  # input A already has the self-loop

    # 2. DTilde: the degree matrix of ATilde
    DTilde = ATilde.sum(axis=2)
    DTilde[DTilde <= 0] = 1  # if DTilde <= 0, keep its previous value

    # 3. normalize
    # http://tkipf.github.io/graph-convolutional-networks/
    # paper：normA = DTilde^{-1/2} * ATilde *  DTilde^{-1/2}
    # code in practice：normA = DTilde^{-1} * ATilde
    # DInv = DTilde^{-1}
    DInv = torch.pow(DTilde, -1)
    DInv[torch.isinf(DInv)] = 0.
    DInvMat = torch.stack([torch.diag(x) for x in DInv], axis=0)
    normA = torch.bmm(DInvMat, ATilde)
    return normA


class HoGCNLayer(BasicModule):
    """
    single HoGCN layer
    """
    def __init__(self, inputDim, outputDim):
        """
        inputDim: input channels of nodes
        outputDim: the output channels of nodes
        """
        super(HoGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(inputDim, outputDim))
        self.bias = nn.Parameter(torch.FloatTensor(outputDim))
        # initialization
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)

        self.BN = nn.BatchNorm1d(outputDim)
        self.activition = nn.ReLU()
        

    def forward(self, inputs):
        """
        normA: the normalized adjacency matrix, shape: [B, maxLength, maxLength]
        X: features matrix of nodes, shape: [B, maxLength, C]
        """
        normA, X = inputs
        
        # HoGCN: X' = σ(AXW+B)
        # AX
        AX = torch.bmm(normA, X)
        
        # AXW
        output = torch.einsum('bnd,df->bnf', (AX, self.weight))
        output += self.bias
        output = self.activition(self.BN(output.permute(0, 2, 1))).permute(0, 2, 1)

        return (normA, output)


class HoGCN(BasicModule):
    def __init__(self, inputDim, outputDim, isClassify=True):
        """
        inputDim: input channels of nodes
        hiddenDim: the hidden unit number of GCN
        outputDim: the output channels of nodes
        isClassifiy: if true, output the classification results; else, output the features of nodes
        """
        super(HoGCN, self).__init__()
        hiddenDim = opt.GCNHiddenDim
        GCNLayersNum = len(hiddenDim)
        self.edgeConnectMode = opt.edgeConnectMode
        self.isClassify = isClassify

        # # HoGCN layers
        self.GCNLayers = nn.Sequential()
        self.GCNLayers.add_module("GCN_0", HoGCNLayer(inputDim, hiddenDim[0]))
        if GCNLayersNum > 1:
            for idx in range(GCNLayersNum-1):
                self.GCNLayers.add_module("GCN_"+str(idx+1), HoGCNLayer(hiddenDim[idx], hiddenDim[idx+1]))
    
        if isClassify:
            self.classifier = nn.Linear(hiddenDim[-1], outputDim)

    def forward(self, inputs):
        """
        adjacency: adjacency matrix, shape: [B, maxLength, maxLength], (value \in [0-1])
        node: features matrix of nodes, shape: [B, maxLength, C]
        """
        adjacency, node, _ = inputs
        
        # 1. calculate the weighted adjacency matrix
        A = calAdjacencyMatrix(adjacency, node, self.edgeConnectMode)

        # 2. normalize
        normA = calNormalizedA(A)
        
        # 3. HoGCN
        _, output = self.GCNLayers((normA, node))

        # 4. classify
        if self.isClassify:
            output = self.classifier(output)

        return output