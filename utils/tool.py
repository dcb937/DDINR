import cv2
import tifffile
import os
import numpy as np
from utils.ReadVTK import readVTK, show3D

def get_type_max(data):
    dtype = data.dtype.name
    if dtype == 'uint8':
        # max = 255
        max = np.max(data)
    elif dtype == 'uint12':
        # max = 4098
        max = np.max(data)
    elif dtype == 'uint16':
        # max = 65535
        max = np.max(data)
    elif dtype == 'float32':
        # max = 65535
        max = np.max(data)            # 修改的，不确定
        # max = 1
    elif dtype == 'float64':
        # max = 65535
        max = np.max(data)            # 修改的，不确定
        # max = 1
    elif dtype == 'int16':
        # max = 65535
        max = np.max(data)
    else:
        raise NotImplementedError
    return max

# 3d->dhwc or thwc 2d->hwc
def read_img(path):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        img = tifffile.imread(path)
        if len(img.shape) == 3:
            img = img[...,None]
        assert len(img.shape)==4                  # 即默认通道数为1 即灰度图像
    elif postfix in ['.png','.jpg']:
        img = cv2.imread(path,-1)
        if len(img.shape) == 2:
            img = img[...,None]
        assert len(img.shape)==3
    elif postfix in ['.vtk']:
        img = readVTK(path)
    else:
        raise NotImplemented
    return img  

def save_img(path, img):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        tifffile.imsave(path, img)
    elif postfix in ['.png','.jpg']:
        cv2.imwrite(path, img)
    elif postfix in ['.vtk']:
        show3D(img, 0, path)
    else:
        raise NotImplemented  

def get_folder_size(folder_path:str):
    total_size = 0
    if os.path.isdir(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    else:
        total_size = os.path.getsize(folder_path)
    return total_size