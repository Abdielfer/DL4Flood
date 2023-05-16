import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchgeo.datasets import RasterDataset, unbind_samples, stack_samples
import torchvision.transforms as transforms
from skimage import transform
from rasterio.plot import show
from torchvision.transforms import functional as TF
from scr import util as U
import rasterio as rio
import random

class customDataSet(Dataset):
    def __init__(self, 
                fullList:os.path,
                inLineTransformation : bool = True,
                validationMode:bool = False):
        super(Dataset, self).__init__()
        self.inLineTransformation = inLineTransformation
        self.img_list, self.mask_list = createImageMaskList(fullList)
        self.validationMode = validationMode

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        with rio.open(img_path, 'r') as image:
            img = image.read()
        with rio.open(mask_path, 'r') as label:
            mask = label.read()
                
        img, mask = self.__Standardization__(img,mask, mean=155, std= 194.2)

        if self.validationMode:
            img = U.imageToTensor(img)
            mask = U.imageToTensor(mask) 
            return img, mask
        
        if self.inLineTransformation:
            img, mask = self.__inlineTranformation__(img,mask)
        
        img = U.imageToTensor(img)
        mask = U.imageToTensor(mask)
        return img, mask

    def _VerifyListsContent(sefl):  ## TODO 
        '''
        verify if imag_list and mask_list have same len and names(img-mask) matchs. 
        '''
        pass 

    def __Standardization__(self,img,mask,mean,std):
        imgStd = (img - mean)/std
        maskStd = (mask - mean)/std
        return imgStd, maskStd

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

def createImageMaskList(imgMaskList:os.path):
    '''
    @imgMaskList: A *csv file containig a pair path of images and masks per line. 
    '''
    img_list = []
    mask_list = []
    with open(imgMaskList, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for _,line in enumerate(reader):
            img_list.append(line[0])
            mask_list.append(line[1])
    if len(img_list)!= len(mask_list):
        raise ValueError("Mismatch between the number of images and masks. You can run customDataSet._VerifyListsContent()")
    return img_list, mask_list 
      
def offlineTransformation(imgMaskList:os.path, savePath:os.path,):
    '''
    Perform permanent transformation to image-mask pair and save a transformed copy of <img> and <mask> in <savePath>.
    The rotated image and mask are saved with the original raster profile for reference only. 
    @imgMaskList: A *.csv file containig a pair path to images-masks per line.
    @return: 
            1- <outputPathList> the list of transformed images-mask pair path.
            2 - the last rotated image-mask pair for reference   
    '''
    outputPathList = []
    ImagSavePath = U.createTransitFolder(savePath, 'image')
    maskSavePath = U.createTransitFolder(savePath, 'mask')
    img_list, mask_list = createImageMaskList(imgMaskList)
    for i, m in zip(img_list, mask_list):
        ## Rotate 180deg
        imgRotData, imaProfile = rotateRaster90_kTime(i,2)
        maskRotData, maskProfile = rotateRaster90_kTime(m,2)
        ## Save
        imagePath = U.addSubstringToName(i, '_rot180', destinyPath= ImagSavePath)
        U.createRaster(imagePath,imgRotData, imaProfile, noData = imaProfile['nodata'])
        maskPath = U.addSubstringToName(m, '_rot180', destinyPath= maskSavePath)
        U.createRaster(maskPath, maskRotData.astype('int'), maskProfile)
        #Create list Image-Mask pair path
        newLine = imagePath + ';' + maskPath
        outputPathList.append(newLine)
    
    scvPath = os.path.join(savePath,'transformedImageMaskPairList.csv')
    U.createCSVFromList(scvPath,outputPathList)
    ## Return the last rotated image-mask pair for reference. 
    return scvPath, imgRotData, maskRotData

def rotateRaster90_kTime(raster, K):
    '''
    This function rotate the raster data, keeping the profile from the source. 
    With axes = (1,2) we aviod change in image shape. 
    NOTE: be carefull with dimentions after rotation. 
    @Expected shape: C;H;W
    @raster: raster path. 
    @k: The k times to rotate the image in anticlockwise. 
    '''
    data,profile = U.readRaster(raster)
    rotatedData = np.rot90(data, k=K, axes=(1,2))    
    return rotatedData, profile

def customDataloader(dataset:customDataSet, args:dict) -> DataLoader:
    '''
    @args: Dic : {'batch_size': 1, 'num_workers': 1,'drop_last': True}
    '''
    customDL = DataLoader(dataset,
                        batch_size = args['batch_size'],
                        shuffle = False,
                        num_workers = args['num_workers'],
                        pin_memory = False, 
                        drop_last = args['drop_last'],
                        )
    
    return customDL

def splitDataset(dataset:customDataSet, proportions = [.9,.1] ,seed:int = 42, )-> customDataSet:
    '''
    ref: https://pytorch.org/docs/stable/data.html# 
    '''
    len = dataset.__len__()
    lengths = [int(p *len) for p in proportions]
    lengths[-1] = len - sum(lengths[:-1])
    generator = torch.Generator().manual_seed(seed)
    train_CustomDS, val_CustomDS = random_split(dataset,lengths,generator=generator)
    return train_CustomDS, val_CustomDS
  

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