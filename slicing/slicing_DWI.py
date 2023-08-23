# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 02:31:41 2023

@author: 
"""

import glob as glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

def read_bvals(bvalf):
    with open(bvalf) as f:
        bvals = next(f).split()
    return np.asarray(bvals).astype("float")

def filter_b0(dir_files, HCP):
    bval_b0 = read_bvals(glob.glob(dir_files + "/{}_3T_DWI_dir95_LR.bval".format(HCP))[0])
    bval_b0 = np.where(bval_b0 < 100)
    bval_b0 = np.asarray(bval_b0).astype("int")[0]
    return bval_b0

dir_main = "dir-to-data"

only_b0 = True # !!! only get b0's

train_12 = [
    "100206",
    "103414",
    "103515",
    "104416",
    "104820",
    "105115",
    "105216",
    "107321",
    "107422",
    "107725",
    "108020",
    "108222"
    ]

train_30 = [
    "108525",
    "108323",
    "108121",
    "107018",
    "106824",
    "106521",
    "106319",
    "106016",
    "105923",
    "105620",
    "105014",
    "104012",
    "103818",
    "103212",
    "103111",
    "103010",
    "102816",
    "102715",
    "102614",
    "102513",
    "102311",
    "102109",
    "102008",
    "101915",
    "101410",
    "101309",
    "101107",
    "101006",
    "100610",
    "100408"
    ]

val = [
       "150019",
        "199958",
        "299760",
        "996782"
       ]

test = [
        "100307",
        "114116",
        "397861",
        "499566",
        "599671",
        "698168",
        "793465",
        "899885"
        ]

#%% Validation
for HCP in val:

    dir_files = dir_main + "/" + HCP
    
    b_iter = filter_b0(dir_files, HCP)
    
    # TOPUP to slices
    topup_files = glob.glob(dir_files + "/dir-to-data/*topup_results.nii*")
    
    for i in range(len(topup_files)):
        topup_file = topup_files[i]
        
        file_name = topup_file.split("/")[-1].split(".")[0]

        save_name = dir_main + "/val/slices_topup/" + file_name
        
        topup_data = nib.load(topup_file).get_fdata()
        
        if not only_b0:
            b_iter = range(topup_data.shape[-1])
        
        for j in tqdm(b_iter):
            save_name2 = save_name + "_volume" + str(j)
            for k in range(topup_data.shape[-2]):
                save_name3 = save_name2 + "_slice" + str(k)
                img = nib.Nifti1Image(topup_data[...,k,j][...,None], np.eye(4))
                nib.save(img, save_name3)
    
    # TOPUP field to slices
    topup_files = glob.glob(dir_files + "/dir-to-data/*topup_fout.nii*")
    
    for i in range(len(topup_files)):
        topup_file = topup_files[i]
        
        file_name = topup_file.split("/")[-1].split(".")[0]
        
        save_name = dir_main + "/val/slices_field_topup/" + file_name
        
        topup_data = nib.load(topup_file).get_fdata()
        topup_data = topup_data*(-0.11232) # !!! depends on the dataset
        
        for j in tqdm(range(topup_data.shape[-1])):
            save_name2 = save_name + "_slice" + str(j)
            img = nib.Nifti1Image(topup_data[...,j,None], np.eye(4))
            nib.save(img, save_name2)
            
    # LR & RL to stacked slices
    LR_files = glob.glob(dir_files + "/dir-to-data/*LR.nii*")
    RL_files = glob.glob(dir_files + "/dir-to-data/*RL.nii*")
    
    assert( len(LR_files) == len(RL_files) )
    
    for i in range(len(LR_files)):
        LR_file = LR_files[i]
        RL_file = RL_files[i]
        
        assert(LR_file != RL_file)
        
        file_name = LR_file.split("/")[-1][:file_name.find("_LR.nii")]

        save_name = dir_main + "/val/slices_like_topup/" + file_name
        
        LR_data = nib.load(LR_file).get_fdata()
        RL_data = nib.load(RL_file).get_fdata()
        data = np.stack([LR_data, RL_data], axis=-1)
        
        if not only_b0:
            b_iter = range(data.shape[-2])
        
        for j in tqdm(b_iter):
            save_name2 = save_name + "_volume" + str(j)
            for k in range(data.shape[-3]):
                save_name3 = save_name2 + "_slice" + str(k)
                img = nib.Nifti1Image(data[...,k,j,:][...,None], np.eye(4))
                nib.save(img, save_name3)
                
    print(HCP)

#%% Testing
for HCP in test:

    dir_files = dir_main + "/" + HCP
    
    b_iter = filter_b0(dir_files, HCP)
    
    # TOPUP to slices
    topup_files = glob.glob(dir_files + "/dir-to-data/*topup_results.nii*")
    
    for i in range(len(topup_files)):
        topup_file = topup_files[i]
        
        file_name = topup_file.split("/")[-1].split(".")[0]

        save_name = dir_main + "/test/slices_topup/" + file_name
        
        topup_data = nib.load(topup_file).get_fdata()
        
        if not only_b0:
            b_iter = range(topup_data.shape[-1])
        
        for j in tqdm(b_iter):
            save_name2 = save_name + "_volume" + str(j)
            for k in range(topup_data.shape[-2]):
                save_name3 = save_name2 + "_slice" + str(k)
                img = nib.Nifti1Image(topup_data[...,k,j][...,None], np.eye(4))
                nib.save(img, save_name3)
    
    # TOPUP field to slices
    topup_files = glob.glob(dir_files + "/dir-to-data/*topup_fout.nii*")
    
    for i in range(len(topup_files)):
        topup_file = topup_files[i]
        
        file_name = topup_file.split("/")[-1].split(".")[0]
        
        save_name = dir_main + "/test/slices_field_topup/" + file_name
        
        topup_data = nib.load(topup_file).get_fdata()
        topup_data = topup_data*(-0.11232) # !!! depends on the dataset
        
        for j in tqdm(range(topup_data.shape[-1])):
            save_name2 = save_name + "_slice" + str(j)
            img = nib.Nifti1Image(topup_data[...,j,None], np.eye(4))
            nib.save(img, save_name2)
            
    # LR & RL to stacked slices
    LR_files = glob.glob(dir_files + "/dir-to-data/*LR.nii*")
    RL_files = glob.glob(dir_files + "/dir-to-data/*RL.nii*")
    
    assert( len(LR_files) == len(RL_files) )
    
    for i in range(len(LR_files)):
        LR_file = LR_files[i]
        RL_file = RL_files[i]
        
        assert(LR_file != RL_file)
        
        file_name = LR_file.split("/")[-1][:file_name.find("_LR.nii")]

        save_name = dir_main + "/test/slices_like_topup/" + file_name
        
        LR_data = nib.load(LR_file).get_fdata()
        RL_data = nib.load(RL_file).get_fdata()
        data = np.stack([LR_data, RL_data], axis=-1)
        
        if not only_b0:
            b_iter = range(data.shape[-2])
        
        for j in tqdm(b_iter):
            save_name2 = save_name + "_volume" + str(j)
            for k in range(data.shape[-3]):
                save_name3 = save_name2 + "_slice" + str(k)
                img = nib.Nifti1Image(data[...,k,j,:][...,None], np.eye(4))
                nib.save(img, save_name3)
                
    print(HCP)
    
#%% Training (12)
for HCP in train_12:

    dir_files = dir_main + "/" + HCP
    
    b_iter = filter_b0(dir_files, HCP)
            
    # LR & RL to stacked slices
    LR_files = glob.glob(dir_files + "/dir-to-data/*LR.nii*")
    RL_files = glob.glob(dir_files + "/dir-to-data/*RL.nii*")
    
    assert( len(LR_files) == len(RL_files) )
    
    for i in range(len(LR_files)):
        LR_file = LR_files[i]
        RL_file = RL_files[i]
        
        assert(LR_file != RL_file)
        
        file_name = LR_file.split("/")[-1][:file_name.find("_LR.nii")]

        save_name = dir_main + "/train/slices_like_topup/" + file_name
        
        LR_data = nib.load(LR_file).get_fdata()
        RL_data = nib.load(RL_file).get_fdata()
        data = np.stack([LR_data, RL_data], axis=-1)
        
        if not only_b0:
            b_iter = range(data.shape[-2])
        
        for j in tqdm(b_iter):
            save_name2 = save_name + "_volume" + str(j)
            for k in range(data.shape[-3]):
                save_name3 = save_name2 + "_slice" + str(k)
                img = nib.Nifti1Image(data[...,k,j,:][...,None], np.eye(4))
                nib.save(img, save_name3)
                
    print(HCP)

#%% Training (42)
for HCP in train_12 + train_30:

    dir_files = dir_main + "/" + HCP
            
    # LR & RL to stacked slices
    LR_files = glob.glob(dir_files + "/dir-to-data/*LR.nii*")
    RL_files = glob.glob(dir_files + "/dir-to-data/*RL.nii*")
    
    assert( len(LR_files) == len(RL_files) )
    
    for i in range(len(LR_files)):
        LR_file = LR_files[i]
        RL_file = RL_files[i]
        
        assert(LR_file != RL_file)
        
        file_name = LR_file.split("/")[-1][:file_name.find("_LR.nii")]

        save_name = dir_main + "/train_42/slices_like_topup/" + file_name
        
        LR_data = nib.load(LR_file).get_fdata()
        RL_data = nib.load(RL_file).get_fdata()
        data = np.stack([LR_data, RL_data], axis=-1)
        
        if not only_b0:
            b_iter = range(data.shape[-2])
        
        for j in tqdm(b_iter):
            save_name2 = save_name + "_volume" + str(j)
            for k in range(data.shape[-3]):
                save_name3 = save_name2 + "_slice" + str(k)
                img = nib.Nifti1Image(data[...,k,j,:][...,None], np.eye(4))
                nib.save(img, save_name3)
                
    print(HCP)
