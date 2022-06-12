import os
import math
import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
from os.path import join
from osgeo import gdal, gdalconst
import sklearn.metrics as skmetrics
from models.CHeGCN import reprojectFeature
from config import opt

import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def setupSeed(seed):
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"set seed {seed}")
    else:
        print("do not set seed")


def getDevice():
    if opt.useGpu and torch.cuda.is_available():
        # use the last GPU by default
        deviceId = opt.deviceId if opt.deviceId != None else torch.cuda.device_count()-1
        device = f"cuda:{deviceId}"
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    return device


def getLrSchedule(optimizer, mode="cosWarmRestartHalf"):
    """
    get the learning rate schedule
    """
    warmUpIter = opt.warmUpEpochs
    TMaxIter = opt.TMaxEpochs
    if mode != "cosWarmRestartHalf":
        info = f"lr mode '{mode}' illegal!"
        warnings.warn(info)

    if opt.model == "CHeGCN":
        lrLambda1 = lambda iter: iter / warmUpIter if  iter < warmUpIter else \
            (np.sqrt(0.5) ** ((iter-warmUpIter) / TMaxIter) * 0.5 * (opt.lrMax * opt.GCNLrRate) * (1.0 + math.cos((iter-warmUpIter) / TMaxIter * math.pi))) / (opt.lrMax * opt.GCNLrRate)
        lrLambda2 = lambda iter: iter / warmUpIter if  iter < warmUpIter else \
            (np.sqrt(0.5) ** ((iter-warmUpIter) / TMaxIter) * 0.5 * (opt.lrMax * opt.ConvLrRate) * (1.0 + math.cos((iter-warmUpIter) / TMaxIter * math.pi))) / (opt.lrMax * opt.ConvLrRate)
        lrLambda3 = lambda iter: iter / warmUpIter if  iter < warmUpIter else \
            (np.sqrt(0.5) ** ((iter-warmUpIter) / TMaxIter) * 0.5 * (opt.lrMax * opt.ResnetLrRate) * (1.0 + math.cos((iter-warmUpIter) / TMaxIter * math.pi))) / (opt.lrMax * opt.ResnetLrRate)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lrLambda1, lrLambda2, lrLambda3]) 
    
    else:
        lrLambda = lambda iter: iter / warmUpIter if  iter < warmUpIter else \
                (np.sqrt(0.5) ** ((iter-warmUpIter) / TMaxIter) * 0.5 * opt.lrMax * (1.0 + math.cos((iter-warmUpIter) / TMaxIter * math.pi))) / opt.lrMax
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrLambda)

    return lrScheduler


def drawLoss(loss, savePath, mode="loss"):
    fig, ax = plt.subplots()
    ax.set_ylabel(f'epoch {mode}')
    ax.set_xlabel('epoch')
    ax.plot(np.arange(len(loss))+1, loss, 'b-')
    if "train" in savePath:
        ax.set_title('train loss curve')
    elif "val" in savePath:
        ax.set_title(f"val {mode} curve")
        ax.set_xticks(np.arange(len(loss)) + 1)
        ax.set_xticklabels((np.arange(len(loss)) + 1) * opt.valStep)
    else:
        pass
    fig.savefig(savePath, dpi=500, bbox_inches='tight')
    plt.close("all")


def drawLr(LrList, LabelList, savePath):
    """
    draw leaning rate curve
    """
    assert len(LrList) == len(LabelList), f"LrList dismatch LabelList!"
    colorList = ["orange", "g", "b", "c"]  # LrList最多三条
    fig, ax = plt.subplots()
    ax.set_title('Learning rate curve')
    ax.set_ylabel('lr')
    ax.set_xlabel('epoch')
    for idx, lrCurve in enumerate(LrList):
        ax.plot(np.arange(len(lrCurve))+1, lrCurve, '-', color=colorList[idx], label=LabelList[idx])
    ax.legend(LabelList, loc=1, fontsize=14)
    fig.savefig(savePath, dpi=500, bbox_inches='tight')
    plt.close("all")


def sameTifFormatCreate(dstfp, reffp, dataArr, saveColorTable=False):
    """
    create dstfp tif file with the same format as reffp
    """
    dataset = gdal.Open(reffp, gdal.GA_ReadOnly)
    if len(dataArr.shape) < 3:
        band_count = 1
        dataArr = dataArr[np.newaxis, :]
    else:
        band_count = dataArr.shape[0]
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    datatype = gdalconst.GDT_Byte

    driver = gdal.GetDriverByName("GTiff")
    out_tif = driver.Create(dstfp, xsize, ysize, band_count, datatype)
    out_tif.SetProjection(dataset.GetProjection())
    out_tif.SetGeoTransform(dataset.GetGeoTransform())

    for index, band in enumerate(dataArr):
        band = np.array(band)
        out_tif.GetRasterBand(index+1).WriteArray(band)
    
    if saveColorTable:
        out_tif.GetRasterBand(1).SetColorTable(dataset.GetRasterBand(1).GetColorTable())

    out_tif.FlushCache()
    out_tif = None


def readTif(imgfp):
    """
    read tif file
    """
    imgDs = gdal.Open(imgfp, gdal.GA_ReadOnly)
    if imgDs is None: raise ValueError(f"Open {imgfp} Failed !!")
    imgData = imgDs.ReadAsArray()
    del imgDs

    # return [H, W, C]
    if len(imgData.shape) == 3:
        imgData = imgData.transpose(1, 2, 0)
    return imgData


def getTifColorTable(imgfp, classNum=5):
    """
    get the colorTable of tif
    """
    dataset = gdal.Open(imgfp, gdal.GA_ReadOnly)
    colorTable = dataset.GetRasterBand(1).GetColorTable()
    colorList = []
    for idx in range(classNum):
        colorList.append(colorTable.GetColorEntry(idx+1)[:3])  # colorTable从1开始
    return colorList


def drawClassificationMap(imgName, predLabel):
    """
    create the classification map of test dataset
    """
    refImgPath = join(f"{opt.parkDir}/slide", imgName)
    testTileIds = os.path.basename(opt.testTileIds)[:-4]
    outputDir = join(opt.outputDir, opt.tag, "classficationTif", f"{opt.testModel[:-4]}_{testTileIds}")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)

    sameTifFormatCreate(
        join(outputDir, imgName),
        refImgPath,
        predLabel,
        saveColorTable=True
    )


def metrics(y_true, yPred):
    accuracy = skmetrics.accuracy_score(y_true, yPred)
    kappa = skmetrics.cohen_kappa_score(y_true, yPred)
    f1_micro = skmetrics.f1_score(y_true, yPred, average="micro")
    f1_macro = skmetrics.f1_score(y_true, yPred, average="macro")
    f1_weighted = skmetrics.f1_score(y_true, yPred, average="weighted")
    recall_micro = skmetrics.recall_score(y_true, yPred, average="micro")
    recall_macro = skmetrics.recall_score(y_true, yPred, average="macro")
    recall_weighted = skmetrics.recall_score(y_true, yPred, average="weighted")
    precision_micro = skmetrics.precision_score(y_true, yPred, average="micro")
    precision_macro = skmetrics.precision_score(y_true, yPred, average="macro")
    precision_weighted = skmetrics.precision_score(y_true, yPred, average="weighted")

    return dict(accuracy=accuracy,
                kappa=kappa,
                f1_micro=f1_micro,
                f1_macro=f1_macro,
                f1_weighted=f1_weighted,
                recall_micro=recall_micro,
                recall_macro=recall_macro,
                recall_weighted=recall_weighted,
                precision_micro=precision_micro,
                precision_macro=precision_macro,
                precision_weighted=precision_weighted,
            )


def accCal(yTruth, yPred, epoch, name):
    outputDir = join(opt.outputDir, opt.tag)
    if name.startswith("test"):
        outputPath = join(outputDir, f"{name}_acc_{opt.testModel}.txt")
    else:
        outputPath = join(outputDir, f"{name}_acc.txt")

    yTruth = torch.cat(yTruth).reshape(-1, 1)
    yPred = torch.cat(yPred).reshape(-1, 1)

    # 1. confusion matrix
    confusionMat = skmetrics.confusion_matrix(yTruth, yPred)

    # 2. all index
    scores = metrics(yTruth, yPred)

    # 3. IoU
    intersection = np.diag(confusionMat)
    union = np.sum(confusionMat, axis=1) + np.sum(confusionMat, axis=0) - np.diag(confusionMat)
    IoU = intersection / union
    scores["IoU_noPark"] = IoU[0]
    scores["IoU_park"] = IoU[1]
    
    # 4. transform indicies to string
    scores_msg = "\n ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
    
    with open(outputPath, 'a', encoding="utf-8") as f:
        if name.startswith("test"):
            f.write(f"test: {opt.testTileIds}\n")
        if epoch:
            f.write(f"[epoch]:{epoch}\n")
        np.savetxt(f, np.array(confusionMat), fmt='%d')
        f.write('confusion matrix \n')
        f.write(scores_msg)
        f.write("\n---------------------\n\n")

    if name.startswith("test"):
        print(name)
        print('confusion matrix \n')
        print(confusionMat)
        print("\n")
        print(scores_msg)
        print("\n")


def trainEpoch(model, device, dataloader, criterion, optimizer):
    """
    train one epoch
    """
    lossEpoch, accEpoch = 0, 0
    yTruth, yPred = [], []

    with tqdm(total=len(dataloader), unit='batch', leave=False, ncols=100, colour="blue") as pbar:
        for idx, batchData in enumerate(dataloader):
            optimizer.zero_grad()

            # 1. data load
            if opt.dataset == "datasetHeGCN":
                adjacency, node, landcover, SLICMap, gts = [x.to(device) for x in batchData[:-1]]
                pred = model((adjacency, node, landcover))
            elif opt.dataset == "datasetCHeGCN":
                adjacency, img, nodeLandcover, SLICMap, gts = [x.to(device) for x in batchData[:-1]]
                pred = model((adjacency, img, nodeLandcover, SLICMap))  # !!!padLen
            else:
                warnings.warn("the dataset must be either datasetHeGCN or datasetCHeGCN!")

            # if HoGCN/HeGCN, transform the node-level result to pixel-level
            if opt.dataset == "datasetHeGCN":
                pred = reprojectFeature(pred, SLICMap)

            # 2. loss calculation
            loss = criterion(pred, gts.long())
            loss.backward()
            optimizer.step()

            # 3. acc calculation
            predLabel = torch.argmax(pred, 1).cpu().to(torch.uint8)
            gts = gts.cpu().to(torch.uint8)
            yTruth.append(gts)
            yPred.append(predLabel)
            acc = (predLabel == gts).sum().item() / torch.numel(gts)
            accEpoch += acc / len(dataloader)
            lossEpoch += loss.item()

            pbar.update(1)
            if (idx+1) % 10 == 0:
                pbar.set_postfix({'loss(batch)': loss.item() / dataloader.batch_size})

    return lossEpoch, accEpoch


def testEpoch(model, device, dataloader, criterion, epoch=None, name='test'):
    """
    test one epoch
    """
    model.eval()
    lossEpoch, accEpoch = 0, 0
    yTruth, yPred = [], []

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit='batch', ncols=100, colour="yellow", leave=False) as pbar:
            for idx, batchData in enumerate(dataloader):
                # 1. data load
                if opt.dataset == "datasetHeGCN":
                    adjacency, node, landcover, SLICMap, gts = [x.to(device) for x in batchData[:-1]]
                    pred = model((adjacency, node, landcover))
                elif opt.dataset == "datasetCHeGCN":
                    adjacency, img, nodeLandcover, SLICMap, gts = [x.to(device) for x in batchData[:-1]]
                    pred = model((adjacency, img, nodeLandcover, SLICMap))
                else:
                    warnings.warn("the dataset must be either datasetHeGCN or datasetCHeGCN!") 

                # if HoGCN/HeGCN, transform the node-level result to pixel-level
                if opt.dataset == "datasetHeGCN":
                    pred = reprojectFeature(pred, SLICMap)

                # 2. loss calculation (for validation)
                loss = criterion(pred, gts.long())
                predLabel = torch.argmax(pred, 1).cpu().to(torch.uint8)
                gts = gts.cpu().to(torch.uint8)
                yTruth.append(gts)
                yPred.append(predLabel)

                # 3. acc calculation
                acc = (predLabel == gts.cpu()).sum().item() / torch.numel(gts)
                accEpoch += acc / len(dataloader)
                lossEpoch += loss.item()

                # 4. draw segmentation results
                if not opt.train:
                    imgName = batchData[-1][0]
                    drawClassificationMap(imgName, predLabel)

                pbar.update(1)
                if (idx+1) % 10 == 0:
                    pbar.set_postfix({'loss(batch)': loss.item() / dataloader.batch_size})

        # acc calculation
        if opt.isAccCal:
            accCal(yTruth, yPred, epoch, name)

    return lossEpoch, accEpoch