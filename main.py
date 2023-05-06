import hydra
from model_set.models import UNetFlood
from scr import util as U
from scr import dataLoader as D
from scr import models_trainer as MT
# from scr import losses as L
from scr.losses import iou_binary,binaryAccuracy, lovasz_hinge 
from omegaconf import DictConfig

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
        
        fullSet = D.customDataSet(cfg['dataPath'])
        trainSet, valSet = D.splitDataset(fullSet)
        args = cfg['dataLoaderArgs']
        train_set = D.customDataloader(trainSet,args)   
        val_set = D.customDataloader(valSet,args)
        model = cfg['model']
        loss_fn = cfg['loss_fn'] 
        losses = []
        OptimizerParams = cfg['OptimizerParams']
        optimizer = cfg['Optimizer']
        optimizer = optimizer(model.parameters(), *OptimizerParams)
        metric = cfg['metric']
        trainer = MT.models_trainer(model,loss_fn,optimizer, metric)
        trainer.set_loaders(train_set,val_set)
        losses = []
        return model, metric, losses
    
@hydra.main(version_base=None, config_path=f"config\OPS", config_name="configPC.yaml")
def main(cfg: DictConfig):
    # ### Performe offline transformations
    # transformer = applyPermanentTransformation(cfg)
    # transformer.transform()

    ### Compute standardizers
    # MinMaxMeanSTD = computeStandardizers(cfg)
    # print(MinMaxMeanSTD)
    
    ## Training cycle
    model, metric, losses = excecuteTraining(cfg)
    name = model.name
    U.saveModel(model, name)

if __name__ == "__main__":
    with U.timeit():
        main()  
