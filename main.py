# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.02.25
@notice  : train and test model
"""


import os
from os.path import join, exists
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import models
import dataset
import warnings
from config import opt
from utils import setupSeed, getDevice, getLrSchedule, trainEpoch, testEpoch, drawLoss, drawLr


def train(**kwargs):
    # update settings
    opt.parse(kwargs)
    # set up random seed
    setupSeed(opt.seed)
    outputDir = join(opt.outputDir, opt.tag)

    # 1. prepare dataset
    trainDataset = getattr(dataset, opt.dataset)(opt.trainTileIds)
    valDataset = getattr(dataset, opt.dataset)(opt.valTileIds)
    trainDataloader = DataLoader(trainDataset, opt.batchSize, shuffle=True, num_workers=opt.numWorkers)
    valDataloader = DataLoader(valDataset, 1)

    # 2. define model
    device = getDevice()
    model = getattr(models, opt.model)(opt.inputDim, 2).to(device)
    
    # output the model structure
    modelOutputPath = join(outputDir, "model.txt")
    modelOutputMode = "a" if exists(modelOutputPath) else "w"
    with open(modelOutputPath, modelOutputMode, encoding="utf-8") as f:
        print(model, file=f)

    # 3. define the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if opt.model == "CHeGCN":
        GCNParams = list(map(id, model.GCN.parameters()))
        ResNetParams = list(map(id, model.resnet.parameters()))
        ConvParams = filter(lambda p: id(p) not in GCNParams + ResNetParams, model.parameters())
        params = [
            {"params": model.GCN.parameters(), "lr": opt.lrMax*opt.GCNLrRate},
            {"params": ConvParams, "lr": opt.lrMax*opt.ConvLrRate},
            {"params": model.resnet.parameters(), "lr": opt.lrMax*opt.ResnetLrRate}
        ]
        optimizer = torch.optim.Adam(params, lr=opt.lrMax, weight_decay=opt.weightDecay)
    elif opt.model in ["HeGCN", "HoGCN"]:
        opt.lrMax = opt.lrMax * opt.GCNLrRate
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lrMax, weight_decay=opt.weightDecay)
    else:
        warnings.warn("model should be HoGCN, HeGCN, or CHeGCN")

    # 4. difine lr schedule
    lrScheduler = getLrSchedule(optimizer, opt.lrMode)

    # 5.start training
    # model save Dir
    checkpointsDir = join(outputDir, "checkpoints")
    if not exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
    if not exists(checkpointsDir):
        os.makedirs(checkpointsDir, exist_ok=True)

    # train loss path and val loss path
    trainLossPath, valLossPath = join(outputDir, "trainLoss.txt"), join(outputDir, "valLoss.txt")
    if not exists(trainLossPath):
        with open(trainLossPath, "w", encoding="utf-8") as f:
            f.write("")
    else:
        print("there is already a trainLoss.txt!")
    if not exists(valLossPath):
        with open(valLossPath, "w", encoding="utf-8") as f:
            f.write("")
    else:
        print("there is already a valLoss.txt!")


    epochs = opt.maxEpoch
    trainLossList, valLossList, valOAList = [], [], []
    lrList = [[] for _ in range(len(optimizer.param_groups))]
    with tqdm(total=epochs, unit='epoch', ncols=120, colour="green") as pbar:
        for epoch in range(epochs):
            # 5.1 train models
            trainLoss, trainAcc = trainEpoch(model, device, trainDataloader, criterion, optimizer)
            
            # 5.2 update lr
            lrScheduler.step()
            for lrGroup in range(len(lrList)):
                lrList[lrGroup].append(optimizer.param_groups[lrGroup]["lr"])

            # 5.3 save the training loss
            trainLossList.append(trainLoss)
            
            # 5.4 save the model
            if (epoch+1) % opt.saveFreq == 0:
                modelPath = join(checkpointsDir, f"epochs_{epoch+1}.pth")
                model.save(optimizer, modelPath)

            # 5.5 update pbar
            pbar.update(1)
            pbar.set_postfix({'lossEpoch': trainLoss, 'accEpoch': trainAcc})
            with open(join(outputDir, "trainLoss.txt"), "a", encoding="utf-8") as f:
                f.write(f"epoch{epoch+1}: lossEpoch_{trainLoss:.6} accEpoch_{trainAcc:.6}\n")

            # 5.6 validate the model
            if (epoch+1) % opt.valStep == 0:
                valLoss, valAcc = testEpoch(model, device, valDataloader, criterion, epoch+1, 'val')
                valLossList.append(valLoss)
                valOAList.append(valAcc)

                with open(join(outputDir, "valLoss.txt"), "a", encoding="utf-8") as f:
                    f.write(f"epoch{epoch+1}: lossEpoch_{valLoss:.6} accEpoch_{valAcc:.6}\n")
                
                # 5.7 draw the training loss and validation loss curve
                drawLoss(trainLossList, join(outputDir, "trainLoss.png"))
                drawLoss(valLossList, join(outputDir, "valLoss.png"))
                drawLoss(valOAList, join(outputDir, "valOA.png"), mode="OA")

                # 5.8 draw lr
                lrLabel = ["GCN", "Conv", "Resnet"] if opt.model == "CHeGCN" else ["lr"]
                drawLr(lrList, lrLabel, join(outputDir, "lrScheduler.png"))


def test(**kwargs):
    # update settings
    opt.parse(kwargs)  # 根据字典kwargs更新config参数

    # 1. prepare dataset
    testDataset = getattr(dataset, opt.dataset)(opt.testTileIds)
    testDataloader = DataLoader(testDataset, 1)

    # 2. define model
    device = getDevice()
    model = getattr(models, opt.model)(opt.inputDim, 2).to(device)

    # 3. load Model
    if opt.testModel:
        testModelPath = join(opt.outputDir, opt.tag, "checkpoints", opt.testModel)
        model.load(testModelPath, None)

    # 4. export meta-path parameters
    if opt.model == "CHeGCN" and opt.isGCNLabel:
        GCNModel = model.GCN
        GCNModel.exportEdgeMetaPathPara()

    # 5. loss
    criterion = torch.nn.CrossEntropyLoss()
    
    # 6. test
    testEpoch(model, device, testDataloader, criterion)

import fire
if __name__ == "__main__":
    fire.Fire()  # annotate it when debug
    
    # mode1. run cmd
    # HoGCN
    # python main.py train --train=True --dataset=datasetHeGCN --model=HoGCN --tag=HoGCN-1 --deviceId=0
    # python main.py test --dataset=datasetHeGCN --model=HoGCN --tag=HoGCN-1 --testModel=epochs_60.pth --deviceId=0

    # HeGCN
    # python main.py train --train=True --dataset=datasetHeGCN --model=HeGCN --tag=HeGCN-1 --deviceId=0
    # python main.py test --dataset=datasetHeGCN --model=HeGCN --tag=HeGCN-1 --testModel=epochs_60.pth --deviceId=0

    # CHoGCN
    # python main.py train --train=True --dataset=datasetCHeGCN --model=CHeGCN --isGCNLabel=False --tag=CHoGCN-1 --deviceId=0
    # python main.py test --dataset=datasetCHeGCN --model=CHeGCN --isGCNLabel=False --tag=CHoGCN-1 --testModel=epochs_60.pth --deviceId=0

    # CHeGCN
    # python main.py train --train=True --dataset=datasetCHeGCN --model=CHeGCN --isGCNLabel=True --tag=CHeGCN-1 --deviceId=0
    # python main.py test --dataset=datasetCHeGCN --model=CHeGCN --isGCNLabel=True --tag=CHeGCN-1 --testModel=epochs_60.pth --deviceId=0

    # mode2. debug
    # CHeGCN
    # train(
    #     train=True,
    #     dataset="datasetCHeGCN",
    #     model="CHeGCN",
    #     isGCNLabel=True,
    #     tag="CHeGCN-1",
    #     deviceId=0
    # )

    # test(
    #     dataset="datasetCHeGCN",
    #     model="CHeGCN",
    #     isGCNLabel=True,
    #     tag="CHeGCN-1",
    #     testModel="epochs_90.pth",
    #     deviceId=0
    # )