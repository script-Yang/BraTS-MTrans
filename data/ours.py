import csv
import os

import logging
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib
import cv2
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from .transforms import build_transforms
from matplotlib import pyplot as plt
from torchvision import transforms
from abc import ABC, abstractmethod

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)
    
# from util.resizer import Resizer
# from util.img_utils import Blurkernel, fft2_m
import fastmri
from scipy.io import loadmat
# @register_operator(name='mri_single')
class MRI_singleOperator(LinearOperator):
    def __init__(self, sampling_rate, device):
        self.device = device
        mask_path = '/data01/home/yangsc/ideayang/medrecon_v2/gen_mask/my_masks/mask_1D_random_x6.mat'
        # mask = loadmat("/data01/home/yangsc/ideayang/resample/ldm_inverse/MASK/mask_1D_random_x6.mat")['mask']
        mask = loadmat(mask_path)['mask']
        mask = torch.Tensor(mask).view(1, 1, 256, 256, 1)
        self.mask = torch.cat([mask, mask], -1).to(device)
        self.sampling_rate = sampling_rate
    def forward(self, data, **kwargs):
        # data bs*3*256*256->bs*3*256*256*2
        data = data.unsqueeze(-1)
        # data_complex = torch.concat([data, torch.zeros_like(data)], dim=-1)
        data_complex = torch.cat([data, torch.zeros_like(data)], dim=-1)
        return fastmri.fft2c(data_complex)*self.mask
    def transpose(self, data, **kwargs):
        # data bs*3*256*256*2
        data_complex = fastmri.ifft2c(data*self.mask)
        return data_complex[:, :, :, :, 0]
    
def normalize(data, mean, stddev, eps=0.0):
    return (data - mean) / (stddev + eps)

def normalize_instance(data, eps=0.0):
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


def complex_abs(data):
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

class OursDataset(Dataset):
    def __init__(
            self,
            root,
            transform,
            challenge,
            sample_rate=1,
            mode='train'
    ):
        self.mode = mode
        # t1_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T1c_test'
        # t2_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T2w_test'
        # t3_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T2f_test'
        # t4_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T1n_test'

        if self.mode == 'train':
            t1_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T1c_test'
            t2_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T2w_test'
            t3_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T2f_test'
            t4_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/T1n_test'
        elif self.mode == 'val'or self.mode == 'test':
            t1_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/small/1'
            t2_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/small/2'
            t3_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/small/3'
            t4_dir = '/data01/home/yangsc/ideayang/LGG_data_skip50/small/4'

        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.t3_dir = t3_dir
        self.t4_dir = t4_dir
        self.size = 256

        self.transform = transforms.Compose([transforms.Resize(self.size)])
        
        self.t1_paths = sorted([os.path.join(self.t1_dir, file) for file in os.listdir(self.t1_dir)])
        self.t2_paths = sorted([os.path.join(self.t2_dir, file) for file in os.listdir(self.t2_dir)])
        self.t3_paths = sorted([os.path.join(self.t3_dir, file) for file in os.listdir(self.t3_dir)])
        self.t4_paths = sorted([os.path.join(self.t4_dir, file) for file in os.listdir(self.t4_dir)])


    def __len__(self):
        # return len(self.examples)
        assert len(self.t1_paths) == len(self.t2_paths) == len(self.t3_paths) == len(self.t4_paths), "Mismatch in modality file counts"
        return len(self.t1_paths)

    def __getitem__(self, i):
        t1_path = self.t1_paths[i]
        t2_path = self.t2_paths[i]
        t3_path = self.t3_paths[i]
        t4_path = self.t4_paths[i]
        # t1= cv2.imread(t1_path, cv2.IMREAD_UNCHANGED)
        # t2 = cv2.imread(t2_path, cv2.IMREAD_UNCHANGED)
        # t3 = cv2.imread(t3_path, cv2.IMREAD_UNCHANGED)
        # t4 = cv2.imread(t4_path, cv2.IMREAD_UNCHANGED)
    
        t1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        t2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        t3 = cv2.imread(t3_path, cv2.IMREAD_GRAYSCALE)
        t4 = cv2.imread(t4_path, cv2.IMREAD_GRAYSCALE)
        
        t1 = torch.tensor(t1).unsqueeze(0).float()/255.
        t2 = torch.tensor(t2).unsqueeze(0).float()/255.
        t3 = torch.tensor(t3).unsqueeze(0).float()/255.
        t4 = torch.tensor(t4).unsqueeze(0).float()/255. 
        t1 = self.transform(t1)
        t2 = self.transform(t2)
        t3 = self.transform(t3)
        t4 = self.transform(t4)
        # t1 = 2.0*t1-1.0
        # t2 = 2.0*t2-1.0
        # t3 = 2.0*t3-1.0
        # t4 = 2.0*t4-1.0

        pd_img = t1
        pdf_img = t2
        op = MRI_singleOperator(sampling_rate=0.5, device=pd_img.device)
        pd_kspace = op.forward(pd_img)
        pd_mask = op.mask
        pd_target = pd_img
        pd_fname = pathlib.Path(t1_path).stem
        pdfs_kspace = op.forward(pdf_img)
        pdfs_mask = op.mask
        pdfs_target = pdf_img
        pdfs_fname = pathlib.Path(t2_path).stem

        # pd_sample = (pd_kspace, pd_mask, pd_target, (2,1), pd_fname, i)
        # pdfs_sample = (pdfs_kspace, pdfs_mask, pdfs_target, (2,1), pdfs_fname, i)
        
        pd_kspace = pd_kspace.squeeze(0).squeeze(0)
        # pd_kspace = pd_kspace.permute(2,0,1)
        pdfs_kspace = pdfs_kspace.squeeze(0).squeeze(0)
        # pdfs_kspace = pdfs_kspace.permute(2,0,1)

        pd_kspace = fastmri.ifft2c(pd_kspace)
        pdfs_kspace = fastmri.ifft2c(pdfs_kspace)
        pd_kspace = complex_abs(pd_kspace)
        pdfs_kspace = complex_abs(pdfs_kspace)

        pd_kspace, mean_pd, std_pd = normalize_instance(pd_kspace, eps=1e-11)
        pd_kspace = pd_kspace.clamp(-6, 6)
        pdfs_kspace, mean_pdfs, std_pdfs = normalize_instance(pdfs_kspace, eps=1e-11)
        pdfs_kspace = pdfs_kspace.clamp(-6, 6)

        pd_target = normalize(pd_target, mean_pd, std_pd, eps=1e-11)
        pdfs_target = normalize(pdfs_target, mean_pdfs, std_pdfs, eps=1e-11)


        pd_sample = (pd_kspace, pd_target, mean_pd, std_pd, pd_fname, i)
        pdfs_sample = (pdfs_kspace, pdfs_target, mean_pdfs, std_pdfs, pdfs_fname, i)
        
        # pd_sample = 

        return (pd_sample, pdfs_sample, i)

def build_dataset(args, mode='train', sample_rate=1):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode)
    return OursDataset(None, transforms, args.DATASET.CHALLENGE,
                        sample_rate=sample_rate, mode=mode)