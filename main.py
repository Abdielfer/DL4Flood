from model_set.models import UNetFlood
from scr import util as U
from scr import dataLoader as D
from scr import models_trainer as MT
from scr import losses as L
from scr.losses import iou_binary,binaryAccuracy
from omegaconf import DictConfig

class excecuteTraining():
    def __init__(self, cfg:DictConfig) -> None:
        
        fullSet = D.customDataSet(cfg['dataPath'])
        trainSet, valSet = D.splitDataset(fullSet)
        args = cfg['dataLoaderArgs']
        train_set = D.customDataloader(trainSet,args)   
        val_set = D.customDataloader(valSet,args)
        loss_fn = L.lovasz_hinge
        OptimizerParams = cfg['OptimizerParams']
        optimizer = cfg['Optimizer']
        model = cfg['model']
        metric = cfg['metric']
        trainer = MT.models_trainer(model,loss_fn,optimizer, metric)
        trainer.set_loaders(train_set,val_set)
        pass
    
    
