from torch.utils.data import Dataset, DataLoader
import os
import glob

'''
Note: Needs three methods __init__, __getitem__, __len__

'''

class customDataLoader(Dataset):
    def __init__(self, folder_path):
        super(customDataLoader, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'image','*.png'))
        self.mask_files = []
        for img_path in img_files:
             self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path))) 

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = use opencv or pil read image using img_path
            label =use opencv or pil read label  using mask_path
            return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)

    
    