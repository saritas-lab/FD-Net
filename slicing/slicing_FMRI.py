# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 03:24:37 2023

@author: 
"""

import glob as glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

def get_savename(file, hcp=None, series_idx=0, slice_idx=0):
    if hcp:
        file_name = hcp + "_" + file.split("/")[-1][:-7] + "_series" + str(series_idx) + "_slice" + str(slice_idx)
    else:
        file_name = file.split("/")[-1][:-10] + "_series" + str(series_idx) + "_slice" + str(slice_idx)
    return file_name

# HCP_list = ["599671", "698168", "793465", "899885"]
# HCP_list = ["100307", "114116", "397861", "499566"]
HCP_list = ["100206", "103414", "104820", "105115"]

zoom_b = True

for HCP in HCP_list:

    dir_files = f"dir-to-data/{HCP}"
    
    print(HCP)
    
    ############################################# rest1
    #(90, 104, 72, 1200)
    dist_rest1_LR_file = glob.glob(dir_files + f"/unprocessed/3T/rfMRI_REST1_LR/{HCP}_3T_rfMRI_REST1_LR.nii.gz")[0]
    dist_rest1_RL_file = glob.glob(dir_files + f"/unprocessed/3T/rfMRI_REST1_RL/{HCP}_3T_rfMRI_REST1_RL.nii.gz")[0]
    
    LR_data = nib.load(dist_rest1_LR_file).get_fdata()[...,0:1]
    RL_data = nib.load(dist_rest1_RL_file).get_fdata()[...,0:1]
    if zoom_b:
        LR_data = zoom(LR_data, [144/90, 168/104, 1, 1])
        RL_data = zoom(RL_data, [144/90, 168/104, 1, 1])
    data = np.stack([LR_data, RL_data], axis=-1)
    save_path = "dir-to-data/fmri/slices/distorted" # !!! "dir-to-data/fmri/slices/train/distorted" for training data
    for i in tqdm(range(1)):
        for j in range(72):
            img = nib.Nifti1Image(data[:,:,j,i,:,None], np.eye(4))
            nib.save(img, save_path + f"/{get_savename(dist_rest1_LR_file, None, i, j)}")
            
    ############################################# emotion
    #(90, 104, 72, 176)
    dist_emotion_LR_file = glob.glob(dir_files + f"/unprocessed/3T/tfMRI_EMOTION_LR/{HCP}_3T_tfMRI_EMOTION_LR.nii.gz")[0]
    dist_emotion_RL_file = glob.glob(dir_files + f"/unprocessed/3T/tfMRI_EMOTION_RL/{HCP}_3T_tfMRI_EMOTION_RL.nii.gz")[0]
 
    LR_data = nib.load(dist_emotion_LR_file).get_fdata()[...,0:1]
    RL_data = nib.load(dist_emotion_RL_file).get_fdata()[...,0:1]
    if zoom_b:
        LR_data = zoom(LR_data, [144/90, 168/104, 1, 1])
        RL_data = zoom(RL_data, [144/90, 168/104, 1, 1])
    data = np.stack([LR_data, RL_data], axis=-1)
    save_path = "dir-to-data/fmri/slices/distorted" # !!! "dir-to-data/fmri/slices/train/distorted" for training data
    for i in tqdm(range(1)):
        for j in range(72):
            img = nib.Nifti1Image(data[:,:,j,i,:,None], np.eye(4))
            nib.save(img, save_path + f"/{get_savename(dist_emotion_LR_file, None, i, j)}")

#%% TOPUP
# HCP_list = ["599671", "698168", "793465", "899885"]
# HCP_list = ["100307", "114116", "397861", "499566"]
HCP_list = ["100206", "103414", "104820", "105115"]

zoom_b = True

i = 0 #!!!

for HCP in HCP_list:

    # dir_files = "dir-to-data/topup"
    dir_files = "dir-to-data/topup_train"
    
    print(HCP)
    
    ############################################# rest1
    #(90, 104, 72)
    topup_i_rest1_file  = glob.glob(dir_files + f"/{HCP}_3T_rfMRI_REST1_0_topup_iout.nii.gz")[0]
    topup_f_rest1_file  = glob.glob(dir_files + f"/{HCP}_3T_rfMRI_REST1_topup_0_fout.nii.gz")[0]
    
    i_data = nib.load(topup_i_rest1_file).get_fdata().mean(axis=-1)
    if zoom_b:
        i_data = zoom(i_data, [144/90, 168/104, 1])
        save_path_i = "dir-to-data/fmri/slices/corrected_topup_resized/image" # !!! "dir-to-data/fmri/slices/train/corrected_topup_resized/image" for training data
    else:
        save_path_i = "dir-to-data/fmri/slices/corrected_topup/image" # !!! "dir-to-data/fmri/slices/train/corrected_topup/image" for training dat
    f_data = nib.load(topup_f_rest1_file).get_fdata()[...,0]
    if zoom_b:
        f_data = zoom(f_data, [144/90, 168/104, 1])
        save_path_f = "dir-to-data/fmri/slices/corrected_topup_resized/field" # !!! "dir-to-data/fmri/slices/train/corrected_topup_resized/field" for training dat
    else:
        save_path_f = "dir-to-data/fmri/slices/corrected_topup/field" # !!! "dir-to-data/fmri/slices/train/corrected_topup/field" for training dat
    for j in range(72):
        i_img = nib.Nifti1Image(i_data[:,:,j,None], np.eye(4))
        nib.save(i_img, save_path_i + f"/{get_savename(topup_i_rest1_file, HCP, i, j)}") 
        f_img = nib.Nifti1Image(f_data[:,:,j,None], np.eye(4))
        nib.save(f_img, save_path_f + f"/{get_savename(topup_f_rest1_file, HCP, i, j)}") 
            
    ############################################# emotion
    #(90, 104, 72, 2)
    topup_i_emotion_file  = glob.glob(dir_files + f"/{HCP}_3T_tfMRI_EMOTION_0_topup_iout.nii.gz")[0]
    topup_f_emotion_file  = glob.glob(dir_files + f"/{HCP}_3T_tfMRI_EMOTION_topup_0_fout.nii.gz")[0]
    
    i_data = nib.load(topup_i_emotion_file).get_fdata().mean(axis=-1)
    if zoom_b:
        i_data = zoom(i_data, [144/90, 168/104, 1])
        save_path_i = "dir-to-data/fmri/slices/corrected_topup_resized/image" # !!! "dir-to-data/fmri/slices/train/corrected_topup_resized/image" for training dat
    else:
        save_path_i = "dir-to-data/fmri/slices/corrected_topup/image" # !!! "dir-to-data/fmri/slices/train/corrected_topup/image" for training dat
    f_data = nib.load(topup_f_emotion_file).get_fdata()[...,0]
    if zoom_b:
        f_data = zoom(f_data, [144/90, 168/104, 1])
        save_path_f = "dir-to-data/fmri/slices/corrected_topup_resized/field" # !!! "dir-to-data/fmri/slices/train/corrected_topup_resized/field" for training dat
    else:
        save_path_f = "dir-to-data/fmri/slices/corrected_topup/field" # !!! "dir-to-data/fmri/slices/train/corrected_topup/field" for training dat
    for j in range(72):
        i_img = nib.Nifti1Image(i_data[:,:,j,None], np.eye(4))
        nib.save(i_img, save_path_i + f"/{get_savename(topup_i_emotion_file, HCP, i, j)}") 
        f_img = nib.Nifti1Image(f_data[:,:,j,None], np.eye(4))
        nib.save(f_img, save_path_f + f"/{get_savename(topup_f_emotion_file, HCP, i, j)}")
