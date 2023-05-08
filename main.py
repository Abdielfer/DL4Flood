import hydra
from model_set.models import UNetFlood
from scr import util as U
from scr import dataLoader as D
from scr import models_trainer as MT
# from scr import losses as L
from scr.losses import iou_binary,binaryAccuracy, lovasz_hinge 
from omegaconf import DictConfig, OmegaConf

class computeStandardizers():
    def __init__(self, cfg:DictConfig) -> None:
        '''
        @filelist: os.path containing the list of rater 
        '''
        self.list = cfg['trainingDataList']
        self.savePath = cfg['standardizerSavePath']
        self.stardardizers = {}
        self.stardardizers = self.compute()
        pass
    
    def compute(self)-> dict:
        standardizer = U.standardizer()
        standardizer.computeGlobalValues(self.list)
        standardizer.saveGlobals(self.savePath)
        return standardizer.getGlobals()

class applyPermanentTransformation():
    def __init__(self,cfg:DictConfig) -> None:
        self.dataSourcePath = cfg['trainingDataList']
        self.dirToSave = cfg['permanentTransformetionSavePath']
        self.transitFolder = U.createTransitFolder(self.dirToSave, folderName = 'transformedTif')
        pass
    def transform(self):
        D.offlineTransformation(self.dataSourcePath,self.transitFolder)

class excecuteTraining():
    def __init__(self, cfg:DictConfig) -> None:
        args = cfg['parameters']
        trainDataSet = D.customDataSet(cfg['trainingDataList'])
        train_DLoader = D.customDataloader(trainDataSet,args['dataLoaderArgs'])   
        valDataSet = D.customDataSet(cfg['validationDataList'])
        val_DLoader = D.customDataloader(valDataSet,args['dataLoaderArgs'])
        testDataSet = D.customDataSet(cfg['testingDataList'])
        test_DLoader = D.createTransformation(testDataSet,args['dataLoaderArgs'])
        model = args['model']
        loss_fn = args['loss_fn']
        optimizer = args['optimizer']
        optimizer = optimizer(model.parameters(),args['optimizerParams'])
        metric = args['metric_fn']
        trainer = MT.models_trainer(model,loss_fn,optimizer, metric)
        trainer.set_loaders(train_DLoader,val_DLoader,test_DLoader)
        trainLosses, valLosses, testLosses = trainer.train(args['epochs'])
        return model, metric, [trainLosses, valLosses, testLosses]
    
@hydra.main(version_base=None, config_path=f"config", config_name="configPC.yaml")
def main(cfg: DictConfig):

    ## Spliting Trn-Val
    trnList, valList = U.splitPerRegion(cfg['rawDataList'])
    U.createCSVFromList(cfg['trainingDataList'], trnList)
    U.createCSVFromList(cfg['validationDataList'],valList)

    # ### Performe offline transformations
    # transformer = applyPermanentTransformation(cfg)
    # transformer.transform()

    ### Compute standardizers
    # MinMaxMeanSTD = computeStandardizers(cfg)
    # print(MinMaxMeanSTD)
    
    ## Training cycle
    # model, metric, losses = excecuteTraining(cfg)
    # name = model.name
    # U.saveModel(model, name)


if __name__ == "__main__":
    with U.timeit():
        main()  
