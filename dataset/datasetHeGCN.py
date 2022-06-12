# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.02.16
@notice  : dataset for HoGCN and HeGCN
"""

import os
import numpy as np
from numpy import load
import torch
from torch.utils import data
from osgeo import gdal
from config import opt


def readTif(imgfp):
    """
    read tif format data
    """
    imgDs = gdal.Open(imgfp, gdal.GA_ReadOnly)
    if imgDs is None: raise ValueError(f"Open {imgfp} Failed !!")
    imgData = imgDs.ReadAsArray()
    del imgDs

    # [C, H, W]
    return imgData


def getDataPath(varDirList, surfix, tileIds):
    """
    get the path list of data (node, adjacency, landcover, park) of samples in tileIdx
    """
    with open(tileIds, "r", encoding="utf-8") as f:
        imgNameList = f.read().split("\n")
    
    output = []
    for idx, varDir in enumerate(varDirList):
        output.append([f"{varDir}/{x + surfix[idx]}" for x in imgNameList])
    
    return output


class datasetHeGCN(data.Dataset):
    def __init__(self, tileIds):
        """
        tileIds: train/test/val samples
        """
        # only select slideEnhance when training
        slideFolder = "slideEnhanced" if opt.isSlide and "train" in tileIds else "slide"
        
        # data to read
        # adjacency: adjacency matrix, shape: [maxLength, maxLength]
        # node: node feature matrix, shape: [maxLength, C]
        # landcover: landcover label of nodes, shape: [maxLength]
        # lcSLICMap: land cover superpixel map, shape: [H, W]
        # park: label of park, shape: [H, W]
        self.varNameList = ["adjacency", "node", "landcover", "lcSLICMap", "park"]
        self.readDataList = [load, load, load, readTif, readTif]
        varDirList = [f"{opt.lcNodeDataRoot}/{slideFolder}/{x}" for x in self.varNameList[:-2]]
        varDirList += [f"{opt.lcSlicRoot}/{slideFolder}", f"{opt.parkDir}/{slideFolder}"]
        surfix = [".npy", ".npy", ".npy", f"_{opt.lcSlicPara}.tif", ".tif"]
        adjacency, node, landcover, lcSLICMap, park = getDataPath(varDirList, surfix, tileIds)

        for var in self.varNameList:
            setattr(self, var, eval(var))

        # max node number
        self.maxLength = opt.maxLength


    def __getitem__(self, index):
        output = []
        for i, varName in enumerate(self.varNameList):
            varPath = getattr(self, varName)[index]
            varArray = self.readDataList[i](varPath)
            output.append(varArray)
        
        # start from 0
        output[3] = output[3] - 1  # lcSLICMap
        imgName = os.path.basename(self.park[index])
        return [torch.from_numpy(x) for x in output] + [imgName]

    def __len__(self):
        return len(self.node)