import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from PIL import Image
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


class TinyScanNetDataset(Dataset):
    def __init__(self, usage='val', dataset_pickle_file='./tiny_scannet.pkl', skip_every_n_image=1):
        super(TinyScanNetDataset, self).__init__()

        self.to_tensor = transforms.ToTensor()

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]
        self.idx = [i for i in range(0, len(self.data_info[0]), skip_every_n_image)]
        self.data_len = len(self.idx)

    def __getitem__(self, index):
        output={}
        img = Image.open(self.data_info[0][index])
        depth = Image.open(self.data_info[1][index])
        normal = Image.open(self.data_info[2][index])

        #To tensor automatically rescales
        output['image']=self.to_tensor(img)

        #Convert depth to float numpy array for scaling
        depth = np.array(depth, dtype='f')
        depth = depth/1000.0
        output['depth']=self.to_tensor(depth)

        #Convert the normal to tensor and multiply by 2 and subtract 1 to get 3D normal tensor
        normal_tensor = 2*self.to_tensor(normal)-1
        output['normal'] = normal_tensor

        return output

    def __len__(self):
        return self.data_len
