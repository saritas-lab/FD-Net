# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


from fdnet_utils import DataGenerator, DataGeneratorFMRI,\
    print_metrics, model_compile

os.chdir("dir-to-code")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
     
import glob as glob
#%% NETWORK
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
lambda_reg = 1e-5

model = model_compile(optimizer=optimizer, reg=lambda_reg)

# model.summary(line_length=150)
#%% LOAD DATA (DWI)
load_path = "dir-to-data"
batch_size = 4

# train
slice_path_train = load_path + "/train/slices_like_topup/*.nii"
topup_path_train = None
field_path_train = None

list_slices_train = glob.glob(slice_path_train)
list_topups_train = None
list_fields_train = None

list_slices_train.sort()

list_train = list_slices_train
list_topup_train = list_topups_train
list_field_train = list_fields_train

dg_train = DataGenerator(
    list_train,
    list_topup_train,
    list_field_train,
    batch_size=batch_size,
    shuffle=True,
    train=True
    )

# val
slice_path_val = load_path + "/val/slices_like_topup/*.nii"
topup_path_val = None
field_path_val = None

list_slices_val = glob.glob(slice_path_val)
list_topups_val = None
list_fields_val = None

list_slices_val.sort()

list_val   = list_slices_val
list_topup_val = list_topups_val
list_field_val = list_fields_val

dg_val   = DataGenerator(
    list_val,
    list_topup_val,
    list_field_val, 
    batch_size=batch_size,
    shuffle=True,
    train=True
    )

# test
slice_path_test = load_path + "/test/slices_like_topup/*.nii"
topup_path_test = load_path + "/test/slices_topup/*.nii"
field_path_test = load_path + "/test/slices_field_topup/*.nii"

list_slices_test = glob.glob(slice_path_test)
list_topups_test = glob.glob(topup_path_test)
list_fields_test = glob.glob(field_path_test)

list_test  = list_slices_test
list_topup_test = list_topups_test
list_field_test = list_fields_test

list_test.sort()
list_topup_test.sort()
list_field_test.sort()

dg_test  = DataGenerator(
    list_test,
    list_topup_test,
    list_field_test,
    batch_size=1,
    shuffle=False
    )

#%% TRAIN (DWI)
epochs = 200
patience = epochs//10

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=5e-6,
    patience=patience,
    verbose=3,
    mode="min",
    baseline=None,
    restore_best_weights=True
    )       

hist = model.fit(
    dg_train,
    epochs=epochs,
    validation_data=dg_val,
    callbacks=[callback]
    )

plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch Number")
plt.legend(["Train", "Val"], loc="upper right")
plt.show()
# plt.savefig("plot-name", dpi=300, bbox_inches="tight")

hist_df = pd.DataFrame(hist.history)
hist_csv_file = load_path + "/file-name.csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

#%% PLOT/SAVE FIGS/ETC.
for j in tqdm(range(len(dg_test))):
    dat = dg_test[j]
    X = dat[0]
    Y, Y1, Y2, Y3, rigid = model.predict(X, verbose=0)
              
    file_name = list_test[j].split('/')[-1]
    
    i = 0 # batch index
           
    #images
    XLR = X[0][i,:,:,0]
    XRL = X[1][i,:,:,0]
    YLR = Y[i,:,:,0,0]
    YRL = Y[i,:,:,0,1]
    topup_image =  dat[1][0][i,:,:,0,3]
    network_image = Y[i,:,:,0,3]
    topup_field = dat[1][0][i,:,:,0,2]
    network_field = Y[i,:,:,0,2]  
    rigid_transform = rigid[i,:]
    
    # ... save etc.

#%% SAVE WEIGHTS
model.save_weights(load_path + "/weights-name")
#%% LOAD WEIGHTS
model.load_weights(load_path + "/weights-name")
#%% PRINT METRICS (DWI)
list_print  = glob.glob(load_path + "/test/slices_like_topup/*.nii")
list_topup_print = glob.glob(load_path + "/test/slices_topup/*.nii")
list_field_print = glob.glob(load_path + "/test/slices_field_topup/*.nii")

dg_test_print  = DataGenerator(
    list_print,
    list_topup_print,
    list_field_print,
    batch_size=1,
    shuffle=False
    )

for j in tqdm(range(len(dg_test_print))):
    dat = dg_test_print[j]
    X = dat[0]
    Y, _, _, _, abc = model.predict(X, verbose=0)
            
    file_name = list_print[j].split('/')[-1]
           
    i = 0 # batch index
    
    #images
    topup_image =  dat[1][0][i,:,:,0,3]
    network_image = Y[i,:,:,0,3]
    topup_field = dat[1][0][i,:,:,0,2]
    network_field = Y[i,:,:,0,2]  
    
    print_metrics(topup_image,
                  topup_field,
                  network_image,
                  network_field,
                  file_name,
                  mask_field=True,
                  ext="DWI")    


#%% LOAD DATA + TRAIN (FMRI)
load_path = "dir-to-data"
batch_size = 4

# train
slice_path_train = load_path + "/fmri/slices/distorted/*.nii"
topup_path_train = None
field_path_train = None

list_slices_train = glob.glob(slice_path_train)
list_topups_train = None
list_fields_train = None

list_slices_train.sort()

list_train = list_slices_train
list_topup_train = list_topups_train
list_field_train = list_fields_train

dg_train = DataGeneratorFMRI(
    list_train,
    list_topup_train,
    list_field_train,
    batch_size=batch_size,
    shuffle=True,
    train=True
    )

# train
epochs = 64
patience = 4

callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=5e-6,
    patience=patience,
    verbose=3,
    mode="min",
    baseline=None,
    restore_best_weights=True
    )       

hist = model.fit(
    dg_train,
    epochs=epochs,
    callbacks=[callback]
    )

plt.figure()
plt.plot(hist.history["loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch Number")
plt.legend(["Fine-tune"], loc="upper right")
plt.show()
# plt.savefig("plot-name-fine-tune", dpi=300, bbox_inches="tight")

hist_df = pd.DataFrame(hist.history)
hist_csv_file = load_path + "/file-name-fine-tune.csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# weights
model.save_weights(load_path + "/weights-name-fine-tune")
# model.load_weights(load_path + "/weights-name-fine-tune")
#%% PRINT METRICS (FMRI)
load_path = "dir-to-data"

list_print  = glob.glob(load_path + "/fmri/slices/distorted/*.nii")
list_topup_print = glob.glob(load_path + "/fmri/slices/corrected_topup_resized/image/*.nii")
list_field_print = glob.glob(load_path + "/fmri/slices/corrected_topup_resized/field/*.nii")

dg_test_print  = DataGeneratorFMRI(
    list_print,
    list_topup_print,
    list_field_print,
    batch_size=1,
    shuffle=False
    )

for j in tqdm(range(len(dg_test_print))):
    dat = dg_test_print[j]
    X = dat[0]
    Y, _, _, _, abc = model.predict(X, verbose=0) # !!! choose appropriate weights
            
    file_name = list_print[j].split('/')[-1]
    
    i = 0 # batch index
    
    #images
    topup_image =  dat[1][0][i,:,:,0,3]
    network_image = Y[i,:,:,0,3]
    topup_field = dat[1][0][i,:,:,0,2]
    network_field = Y[i,:,:,0,2]
    
    print_metrics(topup_image,
                  topup_field,
                  network_image,
                  network_field,
                  file_name,
                  mask_field=True,
                  ext="FMRI") # !!! "FMRI_finetuned" for finetuning weights
