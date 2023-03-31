import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchgeo.datasets import RasterDataset, unbind_samples, stack_samples
import torchvision.transforms as transforms
from skimage import transform
from torchvision.transforms import functional as TF
from scr import util as U
import rasterio as rio
import random

class customDataSet(Dataset):
    def __init__(self, imgMaskList:os.path, inLineTransformation = False, offLineTransformation = False):
        super(Dataset, self).__init__()
        self.imgMaskList = imgMaskList
        self.inLineTransformation = inLineTransformation
        self.offLineTransformation = offLineTransformation
        self.img_list = []
        self.mask_list = []

        with open(self.imgMaskList, "r") as f:
            reader = csv.reader(f, delimiter=";")
            for _,line in enumerate(reader):
                self.img_list.append(line[0])
                self.mask_list.append(line[1])
        if len(self.img_list)!= len(self.mask_list):
            raise ValueError("Mismatch between the number of images and masks. You can run customDataSet._VerifyListsContent()")

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        with rio.open(img_path, 'r') as image:
            img = image.read()
            # print('img shape at read time', img.shape)
        with rio.open(mask_path, 'r') as label:
            mask = label.read()
            # print('mask shape at read time', mask.shape) 
        if self.inLineTransformation:
            img, mask = self.__inlineTranformation__(img,mask)
        img = U.imageToTensor(img)
        # print('img shape at dataLoader delivery time', img.shape)
        mask = U.imageToTensor(mask, 'int64')
        # print('mask shape at dataLoader delivery time', mask.shape)
        return img, mask

    def _VerifyListsContent(sefl):  ## TODO 
        '''
        verify if imag_list and mask_list have same len and names(img-mask) matchs. 
        '''
        pass 

    def __inlineTranformation__(self,imag,mask):
        '''
          inline transformation. 
          We can keep adding more transformations as needed. 
        '''
        # rotation 90deg
        if random.random() > 0.5:
            imag = transform.rotate(imag,90,preserve_range=True)
            mask = transform.rotate(mask, 90,preserve_range=True)
        
        # h_flip
        if random.random() > 0.5:
            imag = np.ascontiguousarray(imag[:, ::-1, ...])
            mask = np.ascontiguousarray(mask[:, ::-1, ...])

        return imag,mask
       
    def __offlineTransformation__(self, ImagSavePath, maskSavePath,imag, mask):
        '''
        Perform some transformation and save transformed <img> and <mask> in <savePath>.
        '''
        ## Rotate 180deg
        img = transform.rotate(imag, 180, preserve_range=True)
        mask = transform.rotate(mask, 180, preserve_range=True)
        
        # Save
        U.saveImag(ImagSavePath, img)
        U.saveImag(maskSavePath, mask)
        

    
# TODO : define *args type  ## 
class customDataloader(DataLoader):
    '''
    Custom DataLoader is useful if you want to pass arguments from config files to a DataLoader instance. 
    For refferences:
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
    
    '''
    def __init__(self, dataset:customDataSet, args:dict):
        super(DataLoader, self).__init__()
        self.customDataloader = DataLoader(dataset, 
                                           batch_size= args['batch_size'], 
                                           shuffle=False,
                                           num_workers= 1,
                                           pin_memory=False, 
                                           drop_last=True,
                                           )
   
    def getDataloader(self):
        return self.customDataloader



## Helper functions 

# "on Going code, not ready yet"
def createTransformation(*args):
    '''
    TODO: Try to create the transformations from a dictionary
    '''
    tranformation = []
    for i in args:
        tranformation.append(i)
        
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return train_transform