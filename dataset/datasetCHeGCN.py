# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.02.20
@notice  : dataset for CHoGCN and CHeGCN
"""


import os
import numpy as np
from numpy import load
from torch.utils import data
import torch
from osgeo import gdal
from config import opt
from .datasetHeGCN import readTif, getDataPath


class datasetCHeGCN(data.Dataset):
    def __init__(self, tileIds):
        """
        tileIds: train/test/val samples
        """
        # only select slideEnhance when training
        slideFolder = "slideEnhanced" if opt.isSlide and "train" in tileIds else "slide"  # 只有训练集才从slideEnhane里选
        
        # data to read
        # adjacency: adjacency matrix, shape: [maxLength, maxLength]
        # img: images, shape: [C, H, W]
        # nodelandcover: landcover label of nodes, shape: [maxLength]
        # lcSLICMap: land cover superpixel map, shape: [H, W]
        # park: label of park, shape: [H, W]
        self.varNameList = ["adjacency", "img", "nodeLandcover", "lcSLICMap", "park"]
        self.readDataList = [load, readTif, load, readTif, readTif]
        varDirList = [
            f"{opt.lcNodeDataRoot}/{slideFolder}/adjacency",
            f"{opt.imgDir}/{slideFolder}",
            f"{opt.lcNodeDataRoot}/{slideFolder}/landcover",
            f"{opt.lcSlicRoot}/{slideFolder}",
            f"{opt.parkDir}/{slideFolder}"
        ]

        varNgtvDirList = []
        surfix = [".npy", ".tif", ".npy", f"_{opt.lcSlicPara}.tif", ".tif"]
        adjacency, img, nodeLandcover, lcSLICMap, park = getDataPath(varDirList, surfix, tileIds)

        for varName in self.varNameList:
            setattr(self, varName, eval(varName))
        
        # max node number
        self.maxLength = opt.maxLength

    def __getitem__(self, index):
        output = []
        for i, varName in enumerate(self.varNameList):
            varPath = getattr(self, varName)[index]
            varArray = self.readDataList[i](varPath)
            output.append(varArray)

        # transfrom img to [0, 1]
        output[1] = (output[1] / 255.0).astype(np.float32)
        # start from 0
        output[3] = output[3] - 1

        imgName = os.path.basename(self.img[index])
        return [torch.from_numpy(x) for x in output] + [imgName]
    
    def __len__(self):
        return len(self.img)