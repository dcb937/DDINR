import os
import cv2
import numpy as np
from utils.tool import get_type_max, save_img
from omegaconf import OmegaConf
import torch
import json
import sys
from tqdm import tqdm
from einops import rearrange, repeat
from utils.ssim import ssim as ssim_calc
from utils.ssim import ms_ssim as ms_ssim_calc
import copy

def cal_iou_acc_pre(data_gt:np.ndarray,data_hat:np.ndarray,thres:float=1):
    hat = np.copy(data_hat)
    gt = np.copy(data_gt)
    hat[data_hat>=thres]=1
    hat[data_hat<thres]=0
    gt[data_gt>=thres]=1
    gt[data_gt<thres]=0
    tp = (gt*hat).sum()
    tn = ((gt+hat)==0).sum()
    fp = ((gt==0)*(hat==1)).sum()
    fn = ((gt==1)*(hat==0)).sum()
    iou = 1.0*tp/(tp+fp+fn)
    acc = 1.0*(tp+tn)/(tp+fp+tn+fn)
    pre = 1.0*tp/(tp+fp)
    return iou, acc, pre

# data_gt, data_hat 都是原始数据，未经过归一化
def cal_psnr(data_gt:np.ndarray, data_hat:np.ndarray, data_range):
    data_gt = np.copy(data_gt)
    data_hat = np.copy(data_hat)
    psnr = []
    for i in range(0, data_range.shape[0]):
        mse = np.mean(np.power(data_gt[:,i]/data_range[i]-data_hat[:,i]/data_range[i],2))
        psnr.append(-10*np.log10(mse))
    return psnr

    # psnr = []
    # data_gt_tensor = torch.tensor(data_gt)
    # data_hat_tensor = torch.tensor(data_hat)
    # mse = torch.mean((data_gt_tensor - data_hat_tensor) ** 2)
    # if mse == 0:
    #     return float('inf')
    # max_pixel = 0.15
    # psnr.append(20 * np.log10(max_pixel / torch.sqrt(mse)).item())
    # return psnr

def eval_performance(points_array, points_value_array, predict_points, predict_points_value):
    assert np.array_equal(points_array, predict_points), "points_array != predict_points"
    max_range = get_type_max(points_value_array)
    points_value_array = points_value_array.astype(np.float32)
    predict_points_value = predict_points_value.astype(np.float32)

    psnr_value = cal_psnr(points_value_array, predict_points_value, max_range)

    return psnr_value