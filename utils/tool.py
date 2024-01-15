import cv2
import tifffile
import os
import sys
import numpy as np


def get_type_max(data):
    # dtype = data.dtype.name
    # if dtype == 'uint8':
    #     # max = 255
    #     max = np.max(data)
    # elif dtype == 'uint12':
    #     # max = 4098
    #     max = np.max(data)
    # elif dtype == 'uint16':
    #     # max = 65535
    #     max = np.max(data)
    # elif dtype == 'float32':
    #     # max = 65535
    #     max = np.max(data)            # 修改的，不确定
    #     # max = 1
    # elif dtype == 'float64':
    #     # max = 65535
    #     max = np.max(data)            # 修改的，不确定
    #     # max = 1
    # elif dtype == 'int16':
    #     # max = 65535
    #     max = np.max(data)
    # else:
    #     raise NotImplementedError
    max = np.max(data, axis=0)
    return max


def save_img(path, img):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        tifffile.imsave(path, img)
    elif postfix in ['.png','.jpg']:
        cv2.imwrite(path, img)
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


if __name__ == "__main__":
    test = np.random.rand(25,3)
    max_values = np.max(test, axis=0)
    aaa = [1,1,1,5].numpy()
    print(max_values.shape[0])
    print(test)
    print(max_values)