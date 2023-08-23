# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, ZeroPadding2D, Conv2D,\
        UpSampling2D, Concatenate, Cropping2D, Dense, Flatten# MaxPooling2D, Dropout, 

import nibabel as nib
import numpy as np
import math

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_fill_holes
from contextlib import redirect_stdout

from skimage.filters import gaussian

from STN_v2 import BilinearInterpolation

class K_UNIT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(K_UNIT, self).__init__(**kwargs)
        self._pi = math.pi

    def call(self, inputs):
        img, fmap = inputs
        return self.k_unit(img, fmap)
       
    def compute_output_shape(self, input_shape):
        return input_shape[0] 
    
    def sinc(self, x):
        x = tf.where(x==0, 1., x)
        pi_x = self._pi * x 
        val = tf.sin(pi_x) / pi_x
        return val
    
    def kernel(self, x, xi):
        with K.name_scope("kernel"):
            y = self.sinc(x-xi)
            return y
    
    @tf.function
    def k_unit(self, im, f):
        with K.name_scope("k_unit"):
            shape = K.shape(im)
            batch_size = shape[0]
            clip_max = tf.cast(shape[1], dtype=tf.float32) #valid interp limit

            _, mesh_j, _, _ = tf.meshgrid(
                tf.range(batch_size),
                tf.range(shape[1]),
                tf.range(shape[2]),
                tf.range(shape[3]),
                indexing="ij"
                )
            mesh_j = tf.cast(mesh_j, dtype=tf.float32)
            
            i_x = f + mesh_j #distorted grid
            i_x = tf.clip_by_value(i_x, 0., clip_max - 1.) #valid interp limit
            
            # modulated difference of grids --> "K-matrix"
            xi = mesh_j[:,tf.newaxis,...]
            xf = i_x[:,:,tf.newaxis,...]
            Kf = self.kernel(xf, xi)
             
            # K-matrix operations
            out = tf.reduce_sum(tf.tile(im[:,:,tf.newaxis,:,:], [1,1,shape[1],1,1]) * Kf, axis=1)

            return out

#%% LOSSES


def l_be(img, avg=True):
    Dx  = -0.5*img[:,:-2,:,:] + 0.5*img[:,2:,:,:]
    Dy  = -0.5*img[:,:,:-2,:] + 0.5*img[:,:,2:,:]
    
    Dx2 = img[:,:-2,:,:] + img[:,2:,:,:] - 2*img[:,1:-1,:,:]
    Dy2 = img[:,:,:-2,:] + img[:,:,2:,:] - 2*img[:,:,1:-1,:]
    
    Dxy = -0.5*Dx[:,:,:-2,:] + 0.5*Dx[:,:,2:,:]
    Dyx = -0.5*Dy[:,:-2,:,:] + 0.5*Dy[:,2:,:,:]
    
    D2all2  = (K.spatial_2d_padding(Dx2, padding=((1, 1), (0, 0))))**2
    D2all2 += (K.spatial_2d_padding(Dy2, padding=((0, 0), (1, 1))))**2
    D2all2 += (K.spatial_2d_padding(Dxy, padding=((1, 1), (1, 1))))**2
    D2all2 += (K.spatial_2d_padding(Dyx, padding=((1, 1), (1, 1))))**2
                 
    loss = K.mean( D2all2 ) if avg else K.sum( D2all2 )
    
    return loss

def l_rigid():
    def loss(y_true, y_pred):
        # identity transformation
        y_ref = tf.cast([1., 0., 0., 0., 1., 0.], dtype=tf.float32)       
        l = K.mean( K.pow( y_pred - y_ref, 2))       
        return l
    return loss

def l_fdnet(reg):
    def loss(y_true, y_pred):
        field     = y_pred[..., 2] 
        
        in_LR = y_true[..., 0]
        in_RL = y_true[..., 1]
        out_LR    = y_pred[..., 0]
        out_RL    = y_pred[..., 1]
        
        loss_similr = 0.5*( K.mean( K.pow(out_LR - in_LR, 2)) +\
                            K.mean( K.pow(out_RL - in_RL, 2)) )
        if reg:        
            valley = K.sum( K.maximum( K.abs(field) - 32, 0.))
            loss_smooth = l_be(field, avg=True) + 1e3*(valley)
            
            l = loss_similr + reg*loss_smooth
        else:
            l = loss_similr
            
        return l
    return loss

class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs,
                 list_topup_IDs,
                 list_IDs_field,
                 dim=(144,168),
                 n_channels=1,
                 batch_size=1,
                 shuffle=True,
                 train=False):
        self.list_IDs = list_IDs                #list of distorted nii
        self.list_topup_IDs = list_topup_IDs    #list of ref topup images
        self.list_IDs_field = list_IDs_field    #list of ref topup fields
        self.dim = dim                          #2d image
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nsamples = len(self.list_IDs)      #number of nii
        self.train = train                      #no need for TOPUP if training
        self.on_epoch_end()
		
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        batch_indexes = self.indexes[
            index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        [X, Y] = self.__data_generation(batch_indexes)

        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        XLR = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        XLR1 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL1 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        XLR2 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL2 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        XLR3 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL3 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        yt = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        yf = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        yt1 = np.empty_like(yt)
        yf1 = np.empty_like(yf)
        
        yt2 = np.empty_like(yt)
        yf2 = np.empty_like(yf)
        
        yt3 = np.empty_like(yt)
        yf3 = np.empty_like(yf)
        
		# Generate data
        for ii in range(batch_indexes.shape[0]):
            # Store sample
            file_id = batch_indexes[ii]
            
			# Load data
            imx = nib.load(self.list_IDs[file_id]).get_fdata() #LR & RL stacked
            
            if not self.train:
                imtopup = nib.load(self.list_topup_IDs[file_id]).get_fdata()
                imtopup[imtopup<0] = 0
                
                # identify the field
                data = self.list_IDs[file_id].split("/")[-1]
                sub_data = data[:data.find("_3T_DWI_dir")]
                dir_data = data[data.find("_3T_DWI_dir")+11:data.find("_volume")]
                # vol_data = data[data.find("_volume")+7:data.find("_slice")]
                slice_data = data[data.find("_slice")+6:data.find(".nii")]
            
            # normalize
            imxmax = imx.max() 
            if imxmax > 0.:
                imx /= imxmax
                if not self.train:
                    imtopup /= imxmax
                    
            # field
            if not self.train:               
                field = [x for x in self.list_IDs_field if
                         sub_data+"_3T_DWI_dir"+dir_data+"_topup_b0_fout_slice"+slice_data+".nii" in x]
                imy = nib.load(field[0]).get_fdata()
            # imy *= -0.11232 # already compensated for in data
            
            xlr = imx[:,:,0,0]
            xrl = imx[:,:,1,0]
            if not self.train:
                ytopup = imtopup[:,:,0]
                yfield = imy[:,:,0]
            
            XLR[ii,:,:,0] = xlr    
            XRL[ii,:,:,0] = xrl
            
            # multiblur
            
            XLR1[ii,:,:,0] = gaussian(xlr, sigma=0.5)
            XRL1[ii,:,:,0] = gaussian(xrl, sigma=0.5)
            
            XLR2[ii,:,:,0] = gaussian(xlr, sigma=1.5)
            XRL2[ii,:,:,0] = gaussian(xrl, sigma=1.5)
            
            XLR3[ii,:,:,0] = gaussian(xlr, sigma=2.5)
            XRL3[ii,:,:,0] = gaussian(xrl, sigma=2.5)
            
            if not self.train:               
                yt[ii,:,:,0] = imtopup[:,:,0]
                yf[ii,:,:,0] = imy[:,:,0]
                
                yt1[ii,:,:,0] = gaussian(ytopup, sigma=0.5)
                yf1[ii,:,:,0] = gaussian(yfield, sigma=0.5)
                
                yt2[ii,:,:,0] = gaussian(ytopup, sigma=1.5)
                yf2[ii,:,:,0] = gaussian(yfield, sigma=1.5)
                
                yt3[ii,:,:,0] = gaussian(ytopup, sigma=2.5)
                yf3[ii,:,:,0] = gaussian(yfield, sigma=2.5)
        
        # prepare data for outputting
        XLR  = np.nan_to_num(XLR)
        XRL  = np.nan_to_num(XRL)
        XLR1 = np.nan_to_num(XLR1)
        XRL1 = np.nan_to_num(XRL1)
        XLR2 = np.nan_to_num(XLR2)
        XRL2 = np.nan_to_num(XRL2)
        XLR3 = np.nan_to_num(XLR3)
        XRL3 = np.nan_to_num(XRL3)
        yt    = np.nan_to_num(yt)
        yf    = np.nan_to_num(yf)
        yt1   = np.nan_to_num(yt1)
        yf1   = np.nan_to_num(yf1)
        yt2   = np.nan_to_num(yt2)
        yf2   = np.nan_to_num(yf2)
        yt3   = np.nan_to_num(yt3)
        yf3   = np.nan_to_num(yf3)
        trn  = np.array([1., 0., 0., 0., 1., 0.])
        
        XLR     = tf.cast(XLR,      dtype=tf.float32)
        XRL     = tf.cast(XRL,      dtype=tf.float32)
        XLR1    = tf.cast(XLR1,     dtype=tf.float32)
        XRL1    = tf.cast(XRL1,     dtype=tf.float32)
        XLR2    = tf.cast(XLR2,     dtype=tf.float32)
        XRL2    = tf.cast(XRL2,     dtype=tf.float32)
        XLR3    = tf.cast(XLR3,     dtype=tf.float32)
        XRL3    = tf.cast(XRL3,     dtype=tf.float32)
        yt      = tf.cast(yt,       dtype=tf.float32)
        yf      = tf.cast(yf,       dtype=tf.float32)
        yt1     = tf.cast(yt1,      dtype=tf.float32)
        yf1     = tf.cast(yf1,      dtype=tf.float32)
        yt2     = tf.cast(yt2,      dtype=tf.float32)
        yf2     = tf.cast(yf2,      dtype=tf.float32)
        yt3     = tf.cast(yt3,      dtype=tf.float32)
        yf3     = tf.cast(yf3,      dtype=tf.float32)
        trn     = tf.cast(trn,      dtype=tf.float32)
                
        return [XLR, XRL], [tf.stack([XLR,  XRL,  yf,  yt ], axis=-1),
                            tf.stack([XLR1, XRL1, yf1, yt1], axis=-1),
                            tf.stack([XLR2, XRL2, yf2, yt2], axis=-1),
                            tf.stack([XLR3, XRL3, yf3, yt3], axis=-1),
                            trn]

class DataGeneratorFMRI(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    
    def __init__(self,
                 list_IDs,
                 list_topup_IDs,
                 list_IDs_field,
                 dim=(144,168),
                 n_channels=1,
                 batch_size=1,
                 shuffle=False,
                 train=False):
        self.list_IDs = list_IDs                #list of distorted nii
        self.list_topup_IDs = list_topup_IDs    #list of ref topup images
        self.list_IDs_field = list_IDs_field    #list of ref topup fields
        self.dim = dim                          #2d image
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nsamples = len(self.list_IDs)      #number of nii
        self.train = train                      #no need for TOPUP if training
        self.on_epoch_end()
		
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        batch_indexes = self.indexes[
            index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        [X, Y] = self.__data_generation(batch_indexes)

        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        XLR = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        XLR1 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL1 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        XLR2 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL2 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        XLR3 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        XRL3 = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        yt = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        yf = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        
        yt1 = np.empty_like(yt)
        yf1 = np.empty_like(yf)
        
        yt2 = np.empty_like(yt)
        yf2 = np.empty_like(yf)
        
        yt3 = np.empty_like(yt)
        yf3 = np.empty_like(yf)
        
		# Generate data
        for ii in range(batch_indexes.shape[0]):
            # Store sample
            file_id = batch_indexes[ii]
            
			# Load data
            imx = nib.load(self.list_IDs[file_id]).get_fdata() #LR & RL stacked
            
            if not self.train:
                imtopup = nib.load(self.list_topup_IDs[file_id]).get_fdata()
                imtopup[imtopup<0] = 0
                
                # identify the field
                data = self.list_IDs[file_id].split("/")[-1]
                sub_data = data[:data.find("_3T")]
                type_data = data[data.find("_3T"):data.find("_series")]
                series_data = data[data.find("_series")+7:data.find("_slice")]
                slice_data = data[data.find("_slice")+6:data.find(".nii")]
            
            # normalize
            imxmax = imx.max()
            if imxmax > 0.:
                imx /= imxmax
                if not self.train:
                    imtopup /= imxmax
            if not self.train:               
                field = [x for x in self.list_IDs_field if
                         sub_data+"_"+sub_data+type_data+"_topup_0_fout_series"+\
                             series_data+"_slice"+slice_data+".nii" in x]
                imy = nib.load(field[0]).get_fdata()
                # Hz to Px conversion
                imy *= -0.0522 # !!! depends on the dataset
            
            xlr = imx[:,:,0,0]
            xrl = imx[:,:,1,0]
            if not self.train:
                ytopup = imtopup[:,:,0]
                yfield = imy[:,:,0]
            
            XLR[ii,:,:,0] = xlr    
            XRL[ii,:,:,0] = xrl
            
            # multiblur
            
            XLR1[ii,:,:,0] = gaussian(xlr, sigma=0.5)
            XRL1[ii,:,:,0] = gaussian(xrl, sigma=0.5)
            
            XLR2[ii,:,:,0] = gaussian(xlr, sigma=1.5)
            XRL2[ii,:,:,0] = gaussian(xrl, sigma=1.5)
            
            XLR3[ii,:,:,0] = gaussian(xlr, sigma=2.5)
            XRL3[ii,:,:,0] = gaussian(xrl, sigma=2.5)
            
            if not self.train:               
                yt[ii,:,:,0] = imtopup[:,:,0]
                yf[ii,:,:,0] = imy[:,:,0]
                
                yt1[ii,:,:,0] = gaussian(ytopup, sigma=0.5)
                yf1[ii,:,:,0] = gaussian(yfield, sigma=0.5)
                
                yt2[ii,:,:,0] = gaussian(ytopup, sigma=1.5)
                yf2[ii,:,:,0] = gaussian(yfield, sigma=1.5)
                
                yt3[ii,:,:,0] = gaussian(ytopup, sigma=2.5)
                yf3[ii,:,:,0] = gaussian(yfield, sigma=2.5)
        
        # prepare data for outputting
        XLR  = np.nan_to_num(XLR)
        XRL  = np.nan_to_num(XRL)
        XLR1 = np.nan_to_num(XLR1)
        XRL1 = np.nan_to_num(XRL1)
        XLR2 = np.nan_to_num(XLR2)
        XRL2 = np.nan_to_num(XRL2)
        XLR3 = np.nan_to_num(XLR3)
        XRL3 = np.nan_to_num(XRL3)
        yt    = np.nan_to_num(yt)
        yf    = np.nan_to_num(yf)
        yt1   = np.nan_to_num(yt1)
        yf1   = np.nan_to_num(yf1)
        yt2   = np.nan_to_num(yt2)
        yf2   = np.nan_to_num(yf2)
        yt3   = np.nan_to_num(yt3)
        yf3   = np.nan_to_num(yf3)
        trn  = np.array([1., 0., 0., 0., 1., 0.])
        
        XLR     = tf.cast(XLR,      dtype=tf.float32)
        XRL     = tf.cast(XRL,      dtype=tf.float32)
        XLR1    = tf.cast(XLR1,     dtype=tf.float32)
        XRL1    = tf.cast(XRL1,     dtype=tf.float32)
        XLR2    = tf.cast(XLR2,     dtype=tf.float32)
        XRL2    = tf.cast(XRL2,     dtype=tf.float32)
        XLR3    = tf.cast(XLR3,     dtype=tf.float32)
        XRL3    = tf.cast(XRL3,     dtype=tf.float32)
        yt      = tf.cast(yt,       dtype=tf.float32)
        yf      = tf.cast(yf,       dtype=tf.float32)
        yt1     = tf.cast(yt1,      dtype=tf.float32)
        yf1     = tf.cast(yf1,      dtype=tf.float32)
        yt2     = tf.cast(yt2,      dtype=tf.float32)
        yf2     = tf.cast(yf2,      dtype=tf.float32)
        yt3     = tf.cast(yt3,      dtype=tf.float32)
        yf3     = tf.cast(yf3,      dtype=tf.float32)
        trn     = tf.cast(trn,      dtype=tf.float32)
                
        return [XLR, XRL], [tf.stack([XLR,  XRL,  yf,  yt ], axis=-1),
                            tf.stack([XLR1, XRL1, yf1, yt1], axis=-1),
                            tf.stack([XLR2, XRL2, yf2, yt2], axis=-1),
                            tf.stack([XLR3, XRL3, yf3, yt3], axis=-1),
                            trn]

def print_metrics(topup_image,
                  topup_field,
                  network_image,
                  network_field,
                  file_name,
                  mask_field=True,
                  ext="DWI"):
    if ext == "DWI":
        file_name = file_name.split(".")[0].split("_")
        subject = file_name[0]
        direction = file_name[3][len("dir"):]
        volume = file_name[4][len("volume"):]
        slicee = file_name[5][len("slice"):]
    elif ext == "FMRI":
        file_name = file_name.split(".")[0].split("_")
        subject = file_name[0]
        direction = file_name[3]
        volume = file_name[4][len("volume"):]
        slicee = file_name[5][len("slice"):]
    else:
        raise ValueError(f"Unsupported value for ext (= {ext}), expected \"DWI\" or \"FMRI\"!")

    #reorient everything
    network_field = np.fliplr((np.rot90(network_field )))
    topup_image   = np.fliplr((np.rot90(topup_image   )))
    topup_field   = np.fliplr((np.rot90(topup_field   )))
    network_image = np.fliplr((np.rot90(network_image )))
    
    #scalar metrics    
    psnr_image_n_vs_t = peak_signal_noise_ratio(
        topup_image, network_image,
        data_range=np.max(topup_image)
        )
    
    ssim_image_n_vs_t = structural_similarity(
        topup_image, network_image,
        multichannel=False,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        data_range=np.max(topup_image)
        )
    
    if mask_field:
        network_field = network_field.copy()
        topup_field = topup_field.copy()
        topup_otsu, mask = median_otsu(topup_image, median_radius=3)
        mask = binary_fill_holes(mask)
        network_field[~mask] = 0.
        topup_field[~mask] = 0.
        
    psnr_field_n_vs_t = peak_signal_noise_ratio(
        topup_field, network_field,
        data_range=np.max(topup_field)
        )
    
    ssim_field_n_vs_t = structural_similarity(
        topup_field, network_field,
        multichannel=False,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        data_range=np.max(topup_field)
        )
        
    with open("print_fdnet_" + ext + ".txt", 'a') as f:
        with redirect_stdout(f):
            print(("{}\t{}\t{}\t{}\t"+
                  "{}\t{}\t"+
                  "{}\t{}").format(
                subject, direction, volume, slicee,
                psnr_image_n_vs_t, psnr_field_n_vs_t,
                ssim_image_n_vs_t, ssim_field_n_vs_t))

def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel
    
    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]
    
    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding="SAME", data_format="NHWC")

def model_compile(optimizer, input_shape=(144,168,1), reg=None):
    
    output_name="out_unet"
    pad_value = ((8, 8), (12, 12))
    
    print("~~~ model ~~~")
    
    input_LR = Input(input_shape, name="input_LR", dtype=tf.float32)
    input_RL = Input(input_shape, name="input_RL", dtype=tf.float32)
    print("inputs shapes: ", input_LR.shape, input_LR.dtype,
          input_RL.shape, input_RL.dtype)
    
    print("~~~ <model> ~~~")


    print("~~~ unet ~~~")
    
    input_unet = Concatenate(axis=-1, name="input_unet")([input_LR,
                                                          input_RL ])
    print("input_unet shape: ", input_unet.shape, input_unet.dtype)
    
    input_unet_zpad = ZeroPadding2D(padding=pad_value, name="zpad")(input_unet)
    print("input_unet_zpad shape", input_unet_zpad.shape, input_unet_zpad.dtype)
    
    conv0 = Conv2D(16, (7, 7), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(input_unet_zpad)
    conv0_1 = Conv2D(16, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv0)
    print("conv0_1 shape", conv0_1.shape, conv0_1.dtype)
    
    conv1 = Conv2D(32, (5, 5), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv0_1)
    conv1_1 = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv1)
    print("conv1_1 shape", conv1_1.shape, conv1_1.dtype)
    
    conv2 = Conv2D(64, (3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv1_1)
    conv2_1 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv2)
    print("conv2_1 shape", conv2_1.shape, conv2_1.dtype)
    
    conv3 = Conv2D(128, (3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv2_1)
    conv3_1 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv3)
    print("conv3_1 shape", conv3_1.shape, conv3_1.dtype)
    
    conv4 = Conv2D(256,  (3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(conv3_1)
    print("conv4 shape", conv4.shape, conv4.dtype)
    
    ####
    
    upconv3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv4)
    upconv3_1 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(upconv3)
    merge3 = Concatenate(axis=-1)([upconv3_1, conv3_1])
    print("merge3 shape", merge3.shape, merge3.dtype)
    
    upconv2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(merge3)
    upconv2_1 = Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(upconv2)
    merge2 = Concatenate(axis=-1)([upconv2_1, conv2_1])
    print("merge2 shape", merge2.shape, merge2.dtype)
    
    upconv1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(merge2)
    upconv1_1 = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(upconv1)
    merge1 = Concatenate(axis=-1)([upconv1_1, conv1_1])
    print("merge1 shape", merge1.shape, merge1.dtype)
    
    upconv0 = UpSampling2D(size=(2, 2), interpolation="bilinear")(merge1)
    upconv0_1 = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(upconv0)
    merge0 = Concatenate(axis=-1)([upconv0_1, conv0_1])
    print("merge0 shape", merge0.shape, merge0.dtype)
    
    convf = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(merge0)
    
    convf_im = Conv2D(1, (5, 5), activation="relu", padding="same", kernel_initializer="glorot_normal")(convf)
    print("convf_im shape", convf_im.shape, convf_im.dtype)
        
    convf_f_0 = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal")(convf)
    convf_f = Conv2D(1, (5, 5), activation=None, padding="same", kernel_initializer="glorot_normal")(convf_f_0)
    print("convf_f shape", convf_f.shape, convf_f.dtype)
    
    output_image = Cropping2D(cropping=pad_value, name=output_name+"_im")(convf_im)
    output_field = Cropping2D(cropping=pad_value, name=output_name+"_f")(convf_f)
    print("output_image and output_field shapes: ",
          output_image.shape, output_image.dtype, 
          output_field.shape, output_field.dtype)
    
    print("~~~ <unet> ~~~")
    
    print("~~~ K-Unit(s) ~~~")
    
    output_LR = K_UNIT(dynamic=False, name="STU_n")(
        [output_image, output_field])
    output_RL = K_UNIT(dynamic=False, name="STU_p")(
        [output_image, -output_field])
    print("outputs shapes: ", output_LR.shape, output_LR.dtype,
          output_RL.shape, output_RL.dtype)
    
    output_image1 = Lambda(lambda x : gaussian_blur(x[0], kernel_size=2, sigma=0.5))([output_image])
    output_field1 = Lambda(lambda x : gaussian_blur(x[0], kernel_size=2, sigma=0.5))([output_field])
    
    output_image2 = Lambda(lambda x : gaussian_blur(x[0], kernel_size=6, sigma=1.5))([output_image])
    output_field2 = Lambda(lambda x : gaussian_blur(x[0], kernel_size=6, sigma=1.5))([output_field])
    
    output_image3 = Lambda(lambda x : gaussian_blur(x[0], kernel_size=10, sigma=2.5))([output_image])
    output_field3 = Lambda(lambda x : gaussian_blur(x[0], kernel_size=10, sigma=2.5))([output_field])
    
    output_LR1 = K_UNIT(dynamic=False, name="STU_n1")(
        [output_image1, output_field1])
    output_RL1 = K_UNIT(dynamic=False, name="STU_p1")(
        [output_image1, -output_field1])
    print("outputs_1 shapes: ", output_LR1.shape, output_LR1.dtype,
          output_RL1.shape, output_RL1.dtype)
    
    output_LR2 = K_UNIT(dynamic=False, name="STU_n2")(
        [output_image2, output_field2])
    output_RL2 = K_UNIT(dynamic=False, name="STU_p2")(
        [output_image2, -output_field2])
    print("outputs_2 shapes: ", output_LR2.shape, output_LR2.dtype,
          output_RL2.shape, output_RL2.dtype)
    
    output_LR3 = K_UNIT(dynamic=False, name="STU_n3")(
        [output_image3, output_field3])
    output_RL3 = K_UNIT(dynamic=False, name="STU_p3")(
        [output_image3, -output_field3])
    print("outputs_3 shapes: ", output_LR3.shape, output_LR3.dtype,
          output_RL3.shape, output_RL3.dtype)
    
    print("~~~ <K-Unit(s)> ~~~")
    
    print("~~~ Rigid Align ~~~")
    
    input_locnet = Concatenate(axis=-1, name="input_locnet")([input_RL,
                                                              output_RL ])
    print("input_locnet shape: ", input_locnet.shape, input_locnet.dtype)
    
    locnet = Conv2D(4, (5,5), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), name="rigid_conv_1")(input_locnet)
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    locnet = Conv2D(8, (5,5), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal", name="rigid_conv_2")(locnet)
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    locnet = Conv2D(16, (5,5), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", kernel_initializer="glorot_normal", name="rigid_conv_3")(locnet)
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    locnet = Flatten(name="rigid_flatten")(locnet)
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    lsz = 32
    locnet = Dense(lsz, activation=tf.keras.layers.LeakyReLU(alpha=0.1), name="rigid_dense_1")(locnet)
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    locnet = Dense(3, kernel_initializer=tf.keras.initializers.zeros(), name="rigid_0")(locnet) #EyeInit(lsz)
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    cos = tf.math.cos(locnet[:,0])
    sin = tf.math.sin(locnet[:,0])
    dx = locnet[:,1]
    dy = locnet[:,2]
    locnet = tf.keras.layers.Lambda(lambda x: x[0], name="rigid")([K.cast_to_floatx(K.stack([cos, -sin, dx, sin, cos, dy], axis=-1))])
    print("~ locnet shape: ", locnet.shape, locnet.dtype)
    
    output_RL = BilinearInterpolation(input_shape[:-1], name="stn_out")([output_RL, locnet])
    output_RL1 = BilinearInterpolation(input_shape[:-1], name="stn_out_1")([output_RL1, locnet])
    output_RL2 = BilinearInterpolation(input_shape[:-1], name="stn_out_2")([output_RL2, locnet])
    output_RL3 = BilinearInterpolation(input_shape[:-1], name="stn_out_3")([output_RL3, locnet])

    print("~~~ <Rigid Align> ~~~")

    print("~~~ Outputs ~~~")

    output_image = K.clip(output_image, 0., 1.)
    output_LR = K.clip(output_LR, 0., 1.)
    output_RL = K.clip(output_RL, 0., 1.)
    
    output_images = Lambda(
        lambda x : K.stack(x, axis=-1), name="full_scale")(
            [output_LR, output_RL, output_field, output_image]
            )
    print("output_images shapes: ", output_images.shape, output_images.dtype)
    
    output_images1 = Lambda(
        lambda x : K.stack(x, axis=-1), name="multiblur_S")(
            [output_LR1, output_RL1, output_field1, output_image1]
            )
    print("output_images1 shapes: ", output_images1.shape, output_images1.dtype)
    
    output_images2 = Lambda(
        lambda x : K.stack(x, axis=-1), name="multiblur_M")(
            [output_LR2, output_RL2, output_field2, output_image2]
            )
    print("output_images2 shapes: ", output_images2.shape, output_images2.dtype)
    
    output_images3 = Lambda(
        lambda x : K.stack(x, axis=-1), name="multiblur_H")(
            [output_LR3, output_RL3, output_field3, output_image3]
            )
    print("output_images3 shapes: ", output_images3.shape, output_images3.dtype)
        
    print("~~~ <Outputs> ~~~")

    
    model = Model(
        inputs=[input_LR, input_RL],
        outputs=[
            output_images,
            output_images1,
            output_images2,
            output_images3,
            locnet
            ]
        )
    
    model.compile(
        loss        ={ "full_scale":l_fdnet(reg),
                        "multiblur_S":l_fdnet(reg*1e1),
                        "multiblur_M":l_fdnet(reg*1e2),
                        "multiblur_H":l_fdnet(reg*1e3),
                        "rigid":l_rigid(),
                            },
        loss_weights={ "full_scale":0.4,
                        "multiblur_S":0.3,
                        "multiblur_M":0.2,
                        "multiblur_H":0.1,
                        "rigid":0.01,
                                  },
        optimizer   =optimizer,
                  )
    
    return model
