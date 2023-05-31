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
        self.initWeigthFunc = instantiate(OmegaConf.create(cfg.parameters['init_weigth']))
        self.initWeigthParams = cfg.parameters['initWeighParams']
        normalize = cfg.parameters['normalize']
        self.trainDataSet = D.customDataSet(cfg['trainingDataList'], normalize= normalize)
        self.train_DLoader = D.customDataloader(self.trainDataSet,args)   
        self.valDataSet = D.customDataSet(cfg['validationDataList'],normalize= normalize, validationMode = True)
        self.val_DLoader = D.customDataloader(self.valDataSet,args)
        self.testDataSet = D.customDataSet(cfg['testingDataList'], normalize= normalize, validationMode = True)
        self.test_DLoader = D.customDataloader(self.testDataSet,args)  
        model = OmegaConf.create(cfg.parameters['model'])
        self.model = instantiate(model)
        self.loss_fn = instantiate(cfg.parameters['loss_fn'])
        criterion = OmegaConf.create(cfg.parameters['optimizer'])
        self.optimizer = instantiate(criterion, params=self.model.parameters())
        self.metric_fn = instantiate(cfg.parameters['metric_fn'])
        
    
    def excecute(self,epochs):
        self.trainer = MT.models_trainer(self.model,self.loss_fn,self.optimizer, self.metric_fn, self.initWeigthFunc,self.initWeigthParams)
        self.trainer.set_loaders(self.train_DLoader,self.val_DLoader,self.test_DLoader)
        trainLosses, valLosses, testLosses = self.trainer.train(epochs)
        self.trainer.plot_losses()
        return self.model, [trainLosses, valLosses, testLosses]

@hydra.main(version_base=None, config_path=f"config", config_name="configPC.yaml")
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
    logging.info(cfg.parameters.init_weigth)
    logging.info(cfg.parameters.loss_fn)
    logging.info(cfg.parameters.optimizer)
    trainer = excecuteTraining(cfg)
    model,_ = trainer.excecute(cfg.parameters['epochs'])
    U.saveModel(model,nameByTime)


if __name__ == "__main__":
    with U.timeit():
        main()  
