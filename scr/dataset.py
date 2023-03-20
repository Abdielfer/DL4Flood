import os
import glob
import csv
from torch.utils.data import Dataset, DataLoader
from torchgeo.datasets import RasterDataset, unbind_samples, stack_samples
import torchvision.transforms as transforms
import util as U
import rasterio as rio

class customDataSet(Dataset):
    def __init__(self, imgMaskList:os.path):
        super(customDataSet, self).__init__()
        self.imgMaskList = imgMaskList
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
        ## To Try:
        # img = rio.rasterio_loader(img_path)
        
        with rio.open(img_path, 'r') as sat_handle:
            img = U.reshape_as_image(sat_handle.read())
            #  metadata = sat_handle.meta
        with rio.open(mask_path, 'r') as label_handle:
            mask = U.reshape_as_image(label_handle.read())
            mask = mask[..., 0]      
        return {"image": img, "mask": mask}

    def _VerifyListsContent(sefl):
        pass 

    
# TODO : define *args type  ## 
class customDataloader(DataLoader):
    def __init__(self, dataset:customDataSet, *args, inlineTranform: transforms = None, 
                 offlineTranform: transforms = None):
        super(customDataloader, self).__init__()
        self.savePath = args.savepath
        self.dataset_purpose = args.datasetPurpous ### trn, val, tst
        self.customDataloader = DataLoader(dataset,)


    def saveImag(imag):
        ## Save the image in self.savePath. Determine if we save tif with rasterio or png standard. 
        pass
    
    def __inlineTranformation__(self,transformer:transforms,imag,label):
        imag = transformer(imag)
        label = transformer(label)
        return imag,label
       
    def __offlineTransformation__(self, savePath, transformer:transforms, imag, label):
        '''
        Perform the transformation in <transformer> and save <image_transformed> and
         <label_transformed> in <savePath>.
        '''
        imag_trans = transformer(imag)
        self.saveImag(imag_trans)
        label_trans = transformer(label_trans)
        self.saveImag(label_trans)
        
    ## to be finished
    def _SetInitsFromFolder(self, folder_root:os.path):
        '''
        In the absence of *.csv with paht to images and labels, we asume both(image and label has same name 
        in diferent subfolder of main pathFolderRoot <folder_path> )
        :param folder_path: Path to the root where <image> and <labels_burned> are located. 
        '''
        self.folder_path = folder_root
        self.img_files = glob.glob(os.path.join(folder_root,'images','*.tif'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_root,'labels_burned',os.path.basename(img_path)))
            self.saveImag()


## Helper functions
def createTransformation(*args):
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