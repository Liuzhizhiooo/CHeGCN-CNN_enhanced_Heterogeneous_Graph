#coding:utf8
import os
from os.path import join, exists
import warnings
class DefaultConfig(object):
    train = False
    model = 'CHeGCN' # model Name
    backbone = "resnet18"  # pretrained backbone
    isGCNLabel = True  # if true CHeGCN or HeGCN, else CHoGCN or HoGCN
    GCNHiddenDim = [32, 32, 32]  # the hidden units of GCN
    edgeConnectMode = "Geo"  # connection mode, either "Geo" or "Full" (non-local models)
    tag = model # output tag
    
    dataRoot = './dataset/'
    dataset = "datasetCHeGCN"  # dataset
    outputDir = './outputs'
    isSlide = True  # slide Enhance
    slideFoder = "slideEnhanced" if isSlide else "slide"
    lcSlicPara = "n50_c10"
    lcNodeDataRoot = join(dataRoot, f"npy/{lcSlicPara}")  # node data dir
    lcSlicRoot = join(dataRoot, f"SLICResult/segments")  # land cover SLIC segments dir
    
    imgDir = join(dataRoot, "img")
    parkDir = join(dataRoot, "park")
    landcoverDir = join(dataRoot, "landcover")
    maxLength = 63  # max node number
    lcClassNum = 5  # number of land cover classes

    seed = None  # random seed
    inputDim = 3  # input Channels of images

    trainTileIds = f"./dataset/divide/train.txt"
    valTileIds = f"./dataset/divide/val.txt"
    testTileIds = f"./dataset/divide/test.txt"
    isAccCal = True

    loadModel = None  # trained model path
    testModel = None

    batchSize = 4  # batch size
    useGpu = True  # user GPU or not
    deviceId = None  # None: use the last one by default
    numWorkers = 2
    saveFreq = 10  # model save frequency (unit: epoch)
    valStep = 10  # validate model frequency (unit: epoch)

    maxEpoch = 130
    lrMax = 0.001
    lrMode = "cosWarmRestartHalf"
    isMultiLr = True
    GCNLrRate = 0.6
    ConvLrRate = 0.2
    ResnetLrRate = 0.1
    warmUpEpochs = 30
    TMaxEpochs = 20
    weightDecay = 1e-4


def parse(self, kwargs):
    '''
    update parameters
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)

    paraSaveDir = join(self.outputDir, self.tag)
    if not exists(paraSaveDir):
        os.makedirs(paraSaveDir, exist_ok=True)
    paraSavePath = join(paraSaveDir, "hyperParas.txt")
    if self.train:
        with open(paraSavePath, "w") as f:
            f.write("")

    print('user config:')
    tplt = "{0:>20}\t{1:<10}"
    with open(paraSavePath, "a") as f:
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k!="parse":
                value = str(getattr(self, k))
                print(tplt.format(k, value))
                if self.train:
                    f.write(tplt.format(k, value, chr(12288))+"\n")


DefaultConfig.parse = parse
opt = DefaultConfig()
