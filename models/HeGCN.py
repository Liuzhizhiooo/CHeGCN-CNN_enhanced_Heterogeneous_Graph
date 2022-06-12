# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.02.17
@notice  : HeGCN (Heterogeneous Graph Convolutional Networks)
"""

import os
import torch
import numpy as np
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from .BasicModule import BasicModule
from .HoGCN import HoGCNLayer, calNormalizedA
from config import opt


def createEdgeCalPara(lcClassNum=5):
    """
    get the meta-path parameters a_P, b_P
    lcClassNum: the number of land cover classes, including grass, forest, buildings, road, water
    """
    # a_ij = a_ji in undirected graph
    paraNum = int(lcClassNum * (lcClassNum + 1) / 2)
    linearPara = torch.zeros((2, paraNum, 1))
    # y = ax + b
    linearPara[0] = init.xavier_uniform_(torch.randn((paraNum, 1)))  # a_P
    linearPara[1] = init.constant_(torch.randn((paraNum, 1)), 0)  # b_P
    return nn.Parameter(linearPara)


def calMetaPathAdjacencyMatrix(adjacency, node, landcover, edgeCalPara, edgeConnectMode="Geo"):
    """
    calculate the weighted adjacency matrix with the meta-path correction.
    if nodes v_1 and v_2 are connected; feature vectors are x, y; land cover labels are i, j
    the calculation of the edge weight e_12 between v_1 and v_2:
    e_12 = exp(a_ij * (x^T·y) + b_ij), a_ij=a_P

    node: node feature matrix, shape: [B, maxLength, C]
    adjacency: adjacency matrix, shape: [B, maxLength, maxLength] (values are 0/1)
    landcover: landcover label of nodes, shape: [B, maxLength]
    edgeCalPara: meta-path parameters, shape: [2, paraNum, 1]
    edgeConnectMode: edge connection mode, "Geo" or "Full"(non-local connection)
    """
    maxLength = opt.maxLength

    # normalize node features
    node = F.normalize(node, p=2, dim=2)

    # if non-local networks, all elements in adjacency matrix are set 1
    if edgeConnectMode == "Full":
        adjacency = torch.ones_like(adjacency).to(adjacency.device)

    # get the corresponding meta-path index of landcover labels (i, j) of nodes (v_1, v2)
    landcoverRowMatrix = landcover.unsqueeze(2).repeat((1, 1, maxLength))
    landcoverColMatrix = landcoverRowMatrix.permute(0, 2, 1)
    weightIdxMatrix = ((landcoverRowMatrix + 1) / 2.0 * landcoverRowMatrix + landcoverColMatrix).to(torch.uint8)
    
    # keep the lower triangle of weightIdxMatrix, since the graph is undirected
    weightIdxMatrix = torch.tril(weightIdxMatrix)
    weightIdxMatrix += weightIdxMatrix.permute(0, 2, 1) - torch.stack([torch.diag(x) for x in weightIdxMatrix.diagonal(dim1=1, dim2=2)], dim=0)

    # get the a_p and b_p
    aMatrix = torch.zeros_like(weightIdxMatrix, dtype=torch.float32)
    bMatrix = torch.zeros_like(weightIdxMatrix, dtype=torch.float32)
    for idx in range(len(edgeCalPara[0])):
        idxMatrix = weightIdxMatrix==idx
        template = torch.ones((idxMatrix.sum(), ), dtype=torch.float32).to(weightIdxMatrix.device)
        aMatrix[idxMatrix] = edgeCalPara[0][idx] * template
        bMatrix[idxMatrix] = edgeCalPara[1][idx] * template

    # e_12 = exp(a_ij * (x^T·y) + b_ij) if v_1 and v_1 are connected
    newAdjacency = torch.bmm(node, node.permute(0, 2, 1))
    newAdjacency = aMatrix * newAdjacency + bMatrix
    newAdjacency = torch.exp(newAdjacency)
    newAdjacency = torch.mul(newAdjacency, adjacency)
    
    return newAdjacency


class HeGCN(BasicModule):
    def __init__(self, inputDim, outputDim, isClassify=True):
        """
        inputDim: input channels of nodes
        hiddenDim: the hidden unit number of GCN
        outputDim: the output channels of nodes
        lcClassNum: the number of land cover classes, including grass, forest, buildings, road, water
        isClassifiy: if true, output the classification results; else, output the features of nodes
        """
        super(HeGCN, self).__init__()
        hiddenDim = opt.GCNHiddenDim
        lcClassNum = opt.lcClassNum
        self.isClassify = isClassify
        GCNLayersNum = len(hiddenDim)

        # get meta-path parameters a_P, b_P
        self.edgeConnectMode = opt.edgeConnectMode
        print("edgeConnectMode", self.edgeConnectMode)
        self.edgeCalPara = createEdgeCalPara(lcClassNum)
        
        # HeGCN layers
        self.GCNLayers = nn.Sequential()
        self.GCNLayers.add_module("GCN_0", HoGCNLayer(inputDim, hiddenDim[0]))
        if GCNLayersNum > 1:
            for idx in range(GCNLayersNum-1):
                self.GCNLayers.add_module("GCN_"+str(idx+1), HoGCNLayer(hiddenDim[idx], hiddenDim[idx+1]))

        # classifier
        if isClassify:
            self.classifier = nn.Linear(hiddenDim[-1], outputDim)

    def forward(self, inputs):
        """
        adjacency: adjacency matrix, shape: [B, maxLength, maxLength] (value \in [0-1])
        node: node feature matrix, shape: [B, maxLength, C]
        landcover: landcover label of nodes, shape: [B, maxLength]
        """
        adjacency, node, landcover = inputs

        # 1. calculate the adjacency with meta-path correction
        A = calMetaPathAdjacencyMatrix(adjacency, node, landcover, self.edgeCalPara, self.edgeConnectMode)

        # 2. normalize adjacency matrix
        normA = calNormalizedA(A)
        
        # 3. HeGCN
        _, output = self.GCNLayers((normA, node))

        # 4. classify
        if self.isClassify:
            output = self.classifier(output)

        return output


    def exportEdgeMetaPathPara(self):
        """
        导出权重计算参数a, b
        """
        a, b = self.edgeCalPara.cpu().detach().numpy()
        aMatrix = np.zeros((opt.lcClassNum, opt.lcClassNum), dtype=np.float)
        bMatrix = aMatrix.copy()
        for row in range(opt.lcClassNum):
            for col in range(row+1):
                idx = int((row + 1) / 2.0 * row + col)
                aMatrix[row, col], aMatrix[col, row] = a[idx], a[idx]
                bMatrix[row, col], bMatrix[col, row] = b[idx], b[idx]
        edgeCalParaOutputDir = os.path.join(opt.outputDir, opt.tag, "EdgeCalPara", f"{opt.testModel[:-4]}")
        if not os.path.exists(edgeCalParaOutputDir):
            os.makedirs(edgeCalParaOutputDir, exist_ok=True)
        with open(os.path.join(edgeCalParaOutputDir, "a_P.txt"), 'w', encoding="utf-8") as f:
            np.savetxt(f, aMatrix, fmt='%.3f', delimiter=',')
        with open(os.path.join(edgeCalParaOutputDir, "b_P.txt"), 'w', encoding="utf-8") as f:
            np.savetxt(f, bMatrix, fmt='%.3f', delimiter=',')