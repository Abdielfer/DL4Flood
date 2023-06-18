import hydra 
from hydra.utils import instantiate
#from model_set.models import UNetFlood
from scr import util as U
from scr import dataLoader as D
from scr import models_trainer as MT
from scr import losses as L
#from scr.losses import iou_binary,lovasz_hinge 
from omegaconf import DictConfig, OmegaConf
#from torch.optim import Adam, SGD
# import torch
#from torch.nn import MSELoss
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

class excecuteTraining():
    def __init__(self, cfg:DictConfig):
        args = cfg.parameters['dataLoaderArgs']
        self.initWeightFunc = instantiate(OmegaConf.create(cfg.parameters['init_weight']))
        self.initWeightParams = cfg.parameters['initWeightParams']
        normalize = cfg.parameters['normalize']
        inTransform = cfg.parameters['inlineTransform']
        self.trainDataSet = D.customDataSet(cfg['trainingDataList'], normalize= normalize,inLineTransform= inTransform)
        self.train_DLoader = D.customDataloader(self.trainDataSet,args)   
        self.valDataSet = D.customDataSet(cfg['validationDataList'],normalize= normalize,inLineTransform= inTransform, validationMode = True)
        self.val_DLoader = D.customDataloader(self.valDataSet,args)
        self.testDataSet = D.customDataSet(cfg['testingDataList'], normalize= normalize,inLineTransform= inTransform, validationMode = True)
        self.test_DLoader = D.customDataloader(self.testDataSet,args)  
        model = OmegaConf.create(cfg.parameters['model'])
        self.model = instantiate(model)
        self.loss_fn = instantiate(cfg.parameters['loss_fn'])
        criterion = OmegaConf.create(cfg.parameters['optimizer'])
        self.optimizer = instantiate(criterion, params=self.model.parameters())
        self.metric_fn = instantiate(cfg.parameters['metric_fn'])
        
    
    def excecute(self,epochs):
        self.trainer = MT.models_trainer(self.model,self.loss_fn,self.optimizer, self.metric_fn, self.initWeightFunc,self.initWeightParams)
        self.trainer.set_loaders(self.train_DLoader,self.val_DLoader,self.test_DLoader)
        trainLosses, valLosses, testLosses = self.trainer.train(epochs)
        self.trainer.plot_losses()
        return self.model, [trainLosses, valLosses, testLosses]

def logger(cfg: DictConfig, nameByTime):
    '''
    You can log all you want here!
    '''
    logging.info(f"Model saved as :{nameByTime}")
    logging.info(f"Model: {cfg.parameters.model}")
    logging.info(f"Weight Init: {cfg.parameters.init_weight}")
    logging.info(f"Parameters of weights initialization: {cfg.parameters['initWeightParams']}")
    logging.info(f"Loss: {cfg.parameters.loss_fn}")
    logging.info(f"Optimizer: {cfg.parameters.optimizer}")
    logging.info(f"Metric function: {cfg.parameters.metric_fn}")
    logging.info(f"DataLoader args: {cfg.parameters['dataLoaderArgs']}")        
    logging.info(f"Training DataSet: {cfg['trainingDataList']}")   
    logging.info(f"Validation DataSet: {cfg['validationDataList']}")
    logging.info(f"Test DataSet: {cfg['testingDataList']}")


@hydra.main(version_base=None, config_path=f"config", config_name="configPC.yaml")
def main(cfg: DictConfig):
    nameByTime = U.makeNameByTime()
    # ## Spliting Trn-Val
    # trnList, valList = U.splitPerRegion(cfg['rawDataList'],trnFract = cfg['trnFract'])
    # U.createCSVFromList(cfg['trainingDataList'], trnList)
    # U.createCSVFromList(cfg['validationDataList'],valList)

    ### Performe offline transformations
    # transformer = applyPermanentTransformation(cfg)
    # transformer.transform()

    ### Compute standardizers
    # MinMaxMeanSTD = computeStandardizers(cfg)
    # print(MinMaxMeanSTD)
    
    ## Training cycle
    logger(cfg,nameByTime)
    trainer = excecuteTraining(cfg)
    model,_ = trainer.excecute(cfg.parameters['epochs'])
    U.saveModel(model,nameByTime)


if __name__ == "__main__":
    with U.timeit():
        main()  
