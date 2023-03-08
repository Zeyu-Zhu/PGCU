import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.funcation import *
    
    
# preprocess data in different dataset
def Preprocess(pan, ms, up_type):
    lrms = ms.resize((int(ms.size[0]/4),int(ms.size[1]/4)), Image.BICUBIC)
    up_ms = upsampling(lrms, ms.size, up_type)
    ms = np.array(ms).transpose(2, 0, 1) / 255
    lrms = np.array(lrms).transpose(2, 0, 1) / 255
    up_ms = np.array(up_ms).transpose(2, 0, 1) / 255
    pan = np.expand_dims(np.array(pan), axis=0) / 255
    return ms, pan, lrms, up_ms, highpass(pan), highpass(lrms)

# dataset
class MyDataset(Dataset):
    
    def __init__(self, root, ms_path, pan_path, up_type):
        super(MyDataset, self).__init__()
        self.root = root
        self.ms_path = ms_path
        self.pan_path = pan_path
        self.ms_list = os.listdir(root+'/'+ms_path)
        self.pan_list = os.listdir(root+'/'+pan_path)
        self.up_type = up_type
        
    def __getitem__(self, index):
        ms = Image.open(self.root+'/'+self.ms_path+'/'+self.ms_list[index])
        pan = Image.open(self.root+'/'+self.pan_path+'/'+self.pan_list[index])
        ms, pan, lrms, up_ms, hpan, hlrms = Preprocess(pan, ms, self.up_type)
        return ms, pan, lrms, up_ms, hpan, hlrms
    
    def __len__(self):
        return len(self.ms_list)