import hydra
import hydra._internal.instantiate._instantiate2 as instantiate
from model_set.models import UNetFlood
from scr import util as U
from scr import dataLoader as D
from scr import models_trainer as MT
from scr import losses as L
from scr.losses import iou_binary,binaryAccuracy, lovasz_hinge 
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, SGD
import torch
from torch.nn import MSELoss
from torch.nn.init import kaiming_normal_, kaiming_uniform_
import logging

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

class excecuteTrainingVelum():
    def __init__(self, cfg:DictConfig) -> None:
        trainSetList = cfg['trainingDataList']
        valSetList = cfg['validationDataList']
        testSetList = cfg['testingDataList']

        args = {'batch_size': 1, 'num_workers': 4,'drop_last': True}
        self.trainDataSet = D.customDataSet(trainSetList)
        self.train_DLoader = D.customDataloader(self.trainDataSet,args)   
        self.valDataSet = D.customDataSet(valSetList, validationMode = True)
        self.val_DLoader = D.customDataloader(self.valDataSet,args)
        self.testDataSet = D.customDataSet(testSetList, validationMode = True)
        self.test_DLoader = D.customDataloader(self.testDataSet,args)  
        self.model = UNetFlood(1,1)
        self.loss_fn = lovasz_hinge   #  MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr = 0.01, weight_decay=0.01)
    def excecute(self,epochs):
        trainer = MT.models_trainer(self.model,self.loss_fn,self.optimizer, iou_binary, init_func=kaiming_normal_ , mode='fan_in', nonlinearity='relu')
        trainer.set_loaders(self.train_DLoader,self.val_DLoader,self.test_DLoader)
        trainLosses, valLosses, testLosses = trainer.train(epochs)
        return self.model, [trainLosses, valLosses, testLosses]

class excecuteTraining():
    def __init__(self, cfg:DictConfig) -> dict:
        args = cfg.parameters['dataLoaderArgs']
        self.trainDataSet = D.customDataSet(cfg['trainingDataList'])
        self.train_DLoader = D.customDataloader(self.trainDataSet,args)   
        self.valDataSet = D.customDataSet(cfg['validationDataList'], validationMode = True)
        self.val_DLoader = D.customDataloader(self.valDataSet,args)
        self.testDataSet = D.customDataSet(cfg['testingDataList'], validationMode = True)
        self.test_DLoader = D.customDataloader(self.testDataSet,args)  
        
        model = OmegaConf.create(cfg.parameters.model)
        self.model = instantiate(model)
        loss = OmegaConf.create(cfg.parameters.loss_fn)
        self.loss_fn = instantiate(loss)
        criterion = OmegaConf.create(cfg.parameters.optimizer)
        self.optimizer = instantiate(criterion)
        metric = OmegaConf.create(cfg.parameters.metric_fn)
        self.metric_fn = instantiate(metric)
        self.trainer = MT.models_trainer(self.model,self.loss_fn,self.optimizer, self.metric_fn,init_func=kaiming_normal_ , mode='fan_in', nonlinearity='relu')
        self.trainer.set_loaders(self.train_DLoader,self.val_DLoader,self.test_DLoader)
        trainLosses, valLosses, testLosses = self.trainer.train(cfg.parameters['epochs'])
        return self.model, [trainLosses, valLosses, testLosses]
    
    # def excecute(self,epochs):
        

@hydra.main(version_base=None, config_path=f"config", config_name="configMac.yaml")
def main(cfg: DictConfig):
    # ## Spliting Trn-Val
    # trnList, valList = U.splitPerRegion(cfg['rawDataList'])
    # U.createCSVFromList(cfg['trainingDataList'], trnList)
    # U.createCSVFromList(cfg['validationDataList'],valList)

    # ### Performe offline transformations
    # transformer = applyPermanentTransformation(cfg)
    # transformer.transform()

    ### Compute standardizers
    # MinMaxMeanSTD = computeStandardizers(cfg)
    # print(MinMaxMeanSTD)
    
    ## Training cycle
    nameByTime = U.makeNameByTime()
    logging.info(f"Model saved as :{nameByTime}")
    logging.info(cfg.parameters.model)
    logging.info(cfg.parameters.optimizer)
    model,losses = excecuteTraining(cfg)
    # saveModelPath = U.makePath(cfg['saveModelsPath'],nameByTime)
    U.saveModel(model, nameByTime)


if __name__ == "__main__":
    with U.timeit():
        main()  
