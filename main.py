import hydra
import hydra._internal.instantiate._instantiate2 as instantiate
from model_set.models import UNetFlood
from scr import util as U
from scr import dataLoader as D
from scr import models_trainer as MT
from scr import util as U
from scr import losses as L
from scr.losses import iou_binary,binaryAccuracy, lovasz_hinge 
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, SGD

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
        trainDataSet = D.customDataSet(trainSetList)
        train_DLoader = D.customDataloader(trainDataSet,args)   
        valDataSet = D.customDataSet(valSetList)
        val_DLoader = D.customDataloader(valDataSet,args)
        testDataSet = D.customDataSet(testSetList)
        test_DLoader = D.createTransformation(testDataSet,args)
        
        model = UNetFlood(1,1)
        loss_fn = L.lovasz_hinge
        optimizer = Adam(model.parameters(), lr = 0.001) #SGD(model.parameters(), lr=0.001, momentum=0.9)##  
        
        trainer = MT.models_trainer(model,loss_fn,optimizer, iou_binary)
        trainer.set_loaders(train_DLoader,val_DLoader,test_DLoader)
        
        trainLosses, valLosses, testLosses = trainer.train(2)
        return model, [trainLosses, valLosses, testLosses]
                    

class excecuteTraining():
    def __init__(self, cfg:DictConfig) -> None:
        model = OmegaConf.create(cfg.parameters.model)
        # print(args)
        # trainDataSet = D.customDataSet(cfg['trainingDataList'])
        # train_DLoader = D.customDataloader(trainDataSet,args['dataLoaderArgs'])   
        # valDataSet = D.customDataSet(cfg['validationDataList'])
        # val_DLoader = D.customDataloader(valDataSet,args['dataLoaderArgs'])
        # testDataSet = D.customDataSet(cfg['testingDataList'])
        # test_DLoader = D.createTransformation(testDataSet,args['dataLoaderArgs'])
        model = instantiate(model)
        # loss_fn = instantiate(cfg.model)
        # optimizer = args['optimizer']
        # optimizer = optimizer(model.parameters(),args['optimizerParams'])
        # metric = instantiate(cfg.metric_fn)
        # trainer = MT.models_trainer(model,loss_fn,optimizer, metric)
        # trainer.set_loaders(train_DLoader,val_DLoader,test_DLoader)
        # trainLosses, valLosses, testLosses = trainer.train(args['epochs'])
        # return model, metric, [trainLosses, valLosses, testLosses]
        print("model >>","\n", model )
    
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
    excecuteTrainingVelum(cfg)
    # model,losses = excecuteTrainingVelum(cfg)
    # # model, metric, losses = excecuteTraining(cfg)
    # name = model.name
    # U.saveModel(model, name)


if __name__ == "__main__":
    with U.timeit():
        main()  
