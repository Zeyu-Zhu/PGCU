import numpy as np
import cv2
from PIL import Image


def highpass(x):
    low_pass = cv2.boxFilter(x, -1, (5,5))
    return x - low_pass

def upsampling(lrms, shape, up_type):
    if up_type == 'bicubic':
        return lrms.resize(shape, resample=Image.BICUBIC)
    