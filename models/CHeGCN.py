# -*- coding: utf-8 -*-
# author: liuzhizhi
# time: 2020-02-26
# note: CHeGCN (CNN-enhanced Hetergeneous Graph Convolutional networks)

import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import opt
from tqdm import tqdm

from .BasicModule import BasicModule
from .HoGCN import HoGCN
from .HeGCN import HeGCN

def projectFeature(imgFeature, lcSLICMap):
    """
    project the pixel-level feature map to node-level
    take the mean value of pixels in a superpixel as the node feature
    imgFeature: feature map of image, shape: [Batch, C, H, W]
    lcSLICMap: land cover superpixel map, shape: [B, H, W] (value start from 0)
    """
    batchNum = imgFeature.shape[0]
    # node: [B, C, maxLength] (nodeNum <= maxLength)
    node = torch.zeros((*imgFeature.shape[:2], opt.maxLength), dtype=torch.float32).to(imgFeature.device)
    for batchId in range(batchNum):
        nodeNum = lcSLICMap[batchId].max() + 1
        for nodeId in range(nodeNum):
            tempt = imgFeature[batchId][:, lcSLICMap[batchId] == nodeId]
            temptMean = torch.mean(tempt, axis=1)
            node[batchId][:, nodeId] = temptMean
    # node: [B, maxLength, C]
    return node.permute(0, 2, 1)  


def reprojectFeature(nodeFeature, lcSLICMap):
    """
    reproject the node-level features to pixel-level
    the feature of all the pixels in a superpixel is the same after reprojection
    nodeFeature: the features of all nodes, shape: [B, N, C]
    lcSLICMap: land cover superpixel map, shape: [B, H, W] (value start from 0)
    """
    batchNum = nodeFeature.shape[0]
    # imgFeature: [B, C, H, W]
    imgFeature = torch.zeros((batchNum, nodeFeature.shape[2], *lcSLICMap.shape[1:]), dtype=torch.float32).to(nodeFeature.device)
    for batchId in range(batchNum):
        nodeNum = lcSLICMap[batchId].max() + 1
        for nodeId in range(nodeNum):
            idxMatrix = lcSLICMap[batchId] == nodeId
            reprojFeature = nodeFeature[batchId][nodeId].unsqueeze(1).repeat((1, idxMatrix.sum()))
            imgFeature[batchId][:, idxMatrix] = reprojFeature
    return imgFeature


def downloadModel(url, fname):
    """
    download the pretrained model
    url: download url
    fname: save path
    """
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed downloading url %s"%url)
    total_length = r.headers.get('content-length')
    with open(fname, 'wb') as f:
        if total_length is None: # no content length header
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        else:
            total_length = int(total_length)
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                            total=int(total_length / 1024. + 0.5),
                            unit='KB', unit_scale=False, dynamic_ncols=True):
                f.write(chunk)


class CHeGCN(BasicModule):
    """
    CHeGCN (CNN-enhanced Hetergeneous Graph Convolutional networks) model
    """
    def __init__(self, inputDim, outputDim):
        super(CHeGCN, self).__init__()
        # load pretrained backbone
        modelUrl = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        modelDir = "./models/resnet/"
        if not os.path.exists(modelDir):
            os.makedirs(modelDir, exist_ok=True)
        modelPath = f"{modelDir}/{modelUrl.split('/')[-1]}"
        if not os.path.exists(modelPath):
            print("download pretrained backbone ...")
            downloadModel(modelUrl, modelPath)
        self.resnet = getattr(models, opt.backbone)(pretrained=False)
        self.resnet.load_state_dict(torch.load(modelPath))

        # reduce pretrained CNN features
        reduceDim = opt.GCNHiddenDim[-1]
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, reduceDim, 1),
            nn.BatchNorm2d(reduceDim),
            nn.ReLU()
        )

        # GCN model
        if opt.isGCNLabel:
            self.GCN = HeGCN(
                inputDim=reduceDim,
                outputDim=outputDim,
                isClassify=False
            )
        else:
            self.GCN = HoGCN(
                inputDim=reduceDim,
                outputDim=outputDim,
                isClassify=False
            )

        # classifier
        self.outConv = nn.Conv2d(reduceDim, outputDim, kernel_size=1)
    
    def backboneForward(self, x):
        """
        get the feature map of pretrained backbone
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        c1 = self.resnet.layer1(x)
        c2 = self.resnet.layer2(c1)
        return c2

    def forward(self, inputs):
        """
        adjacency: adjacency matrix, shape: [B, maxLength, maxLength]
        x: input images, shape: [B, C, H, W]
        landcover: landcover label of nodes, shape: [B, maxLength]
        lcSLICMap: land cover superpixel map, shape: [B, H, W]
        """
        adjacency, x, landcover, lcSLICMap = inputs
        imsize = x.size()[2:]

        # 1. get pretrained feature map
        # c2: [B, 128, 32, 32]
        c2 = self.backboneForward(x)
        x = F.interpolate(c2, imsize, mode='bilinear', align_corners=True)

        # 2. reduce the pretrained features
        x = self.conv1(x)

        # 3. transform pixels to graph-structure
        node = projectFeature(x, lcSLICMap)

        # 4. Graph convolution
        out = self.GCN((adjacency, node, landcover))

        # 5. transform graph back to pixels
        out = reprojectFeature(out, lcSLICMap)

        # 6. classifier
        logits = self.outConv(out+x)  # feature fusion

        return logits