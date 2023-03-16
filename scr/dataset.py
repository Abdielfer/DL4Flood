import os
import glob
import csv
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import util as U

class customDataSet(Dataset):
    def __init__(self, folder_root:os.path, datasetTipe:str):
        super(customDataSet, self).__init__()
        self.scv_list = U.listFreeFilesInDirByExt(folder_root)  
        self.folder_path = folder_root
        self.img_files = []
        self.mask_files = []
        for scvName in self.scv_list:
            if datasetTipe is scvName:
                file = os.path.join(folder_root,scvName) 
                with open(file, "r") as f:
                    reader = csv.reader(f, delimiter=";")
                    for i, line in enumerate(reader):
                        self.img_files.append(line[0])
                        self.mask_files.append(line[1])
                        
                

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = use opencv or pil read image using img_path
        label =use opencv or pil read label  using mask_path
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __inlineTranformation__(self,transformer:transforms,imag,label):
        imag = transformer(imag)
        label = transformer(label)
        return imag,label
    
    def __offlineTransformation__(self, savePath, transformer:transforms, imag, label):
        '''
        Perform the transformation in <transformer> and save <image_transformed> and
         <label_transformed> in <savePath>.
        '''
        imag = transformer(imag)
        self.saveImag(self.folder_path,imag)
        label = transformer(label)
        self.saveImag(self.folder_path,label)
        return imag,label
    
    def saveImag(self.folder_path,imag):
        pass
    
    def _SetInitsFromFolder(self, folder_root:os.path):
        '''
        In the absence sof *.csv with paht to images and labels, we asume both(image and label has same name in diferent subfolder of main pathFolderRoot <folder_path> )
        :param folder_path: Path to the root where <image> and <labels_burned> are located. 
        '''
        self.folder_path = folder_root
        self.img_files = glob.glob(os.path.join(folder_root,'images','*.tif'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_root,'labels_burned',os.path.basename(img_path)))
             
class customDataloader(DataLoader):
    def __init__(self,dtaset:customDataSet, *args):
        super(customDataloader, self).__init__()

   
