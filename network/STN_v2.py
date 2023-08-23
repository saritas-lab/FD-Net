# -*- coding: utf-8 -*-
"""
https://github.com/oarriaga/STN.keras/blob/dd4dc2051411303777b553e0f8b6b68751941bc5/src/models/layers.py
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

if K.backend() == 'tensorflow':
    import tensorflow as tf

    def K_meshgrid(x, y):
        return tf.meshgrid(x, y)

    def K_linspace(start, stop, num):
        return tf.linspace(start, stop, num)

else:
    raise Exception("Only 'tensorflow' is supported as backend")


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, input_, fmap):
        """
        Args :
          input_ : Input tensor. Its shape should be
              [batch_size, height, width, channel].
              In this implementation, the shape should be fixed for speed.
          new_size : The output size [new_height, new_width]
        ref : 
          http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
        """
        
        shape = tf.shape(input_)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channel = shape[3]
        
        fx = fmap[...,:1]
        fy = fmap[...,-1:]
        
        # batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        height_f = tf.cast(height, dtype=tf.float32)
        width_f = tf.cast(width, dtype=tf.float32)
        # channel_f = tf.cast(channel, dtype=tf.float32)
           
        def _hermite(A, B, C, D, t):
          a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
          b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
          c = A * (-0.5) + C * 0.5
          d = B
          
          return a*t*t*t + b*t*t + c*t + d
          
        def _get_grid_array(n_i, y_i, x_i, c_i):
          n, y, x, c = tf.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
          x = tf.cast(tf.cast(x, dtype=tf.float32) + tf.floor(fx), dtype=tf.int32)
          y = tf.cast(tf.cast(y, dtype=tf.float32) + tf.floor(fy), dtype=tf.int32)
          
          n = tf.expand_dims(n, axis=4)
          y = tf.expand_dims(y, axis=4)
          x = tf.expand_dims(x, axis=4)
          c = tf.expand_dims(c, axis=4)
                    
          return tf.concat([n,y,x,c], axis=4)
          
        def _get_frac_array(y_d, x_d, n, c):
          y = tf.shape(y_d)[0]
          x = tf.shape(x_d)[0]
          y_t = tf.reshape(y_d, [1, -1, 1, 1])
          x_t = tf.reshape(x_d, [1, 1, -1, 1])
          y_t = tf.tile(y_t, (n,1,x,c))
          x_t = tf.tile(x_t, (n,y,1,c))
          
          x_t += fx - tf.floor(fx)
          y_t += fy - tf.floor(fy)
          
          return y_t, x_t
          
        def _get_index_tensor(grid, x, y):
          grid_y = grid[:,:,:,:,1] + y
          grid_x = grid[:,:,:,:,2] + x
          
          grid_y = tf.clip_by_value(grid_y, 0, height-1)
          grid_x = tf.clip_by_value(grid_x, 0, width-1)
          
          new_grid = tf.stack([grid[...,0],
                              grid_y,
                              grid_x,
                              grid[...,3]], axis=-1)
          
          return tf.cast(new_grid, dtype=tf.int32)
          
        n_i = tf.range(batch_size)
        c_i = tf.range(channel)
          
        y_f = tf.linspace(0., height_f-1, height)
        y_i = tf.cast(y_f, dtype=tf.int32)
        y_d = y_f - tf.floor(y_f)
          
        x_f = tf.linspace(0., width_f-1, width)
        x_i = tf.cast(x_f, dtype=tf.int32)
        x_d = x_f - tf.floor(x_f) 
          
        grid = _get_grid_array(n_i, y_i, x_i, c_i)
        y_t, x_t = _get_frac_array(y_d, x_d, batch_size, channel)
          
        i_00 = _get_index_tensor(grid, -1, -1)
        i_10 = _get_index_tensor(grid, +0, -1)
        i_20 = _get_index_tensor(grid, +1, -1)
        i_30 = _get_index_tensor(grid, +2, -1)
            
        i_01 = _get_index_tensor(grid, -1, +0)
        i_11 = _get_index_tensor(grid, +0, +0)
        i_21 = _get_index_tensor(grid, +1, +0)
        i_31 = _get_index_tensor(grid, +2, +0)
            
        i_02 = _get_index_tensor(grid, -1, +1)
        i_12 = _get_index_tensor(grid, +0, +1)
        i_22 = _get_index_tensor(grid, +1, +1)
        i_32 = _get_index_tensor(grid, +2, +1)
            
        i_03 = _get_index_tensor(grid, -1, +2)
        i_13 = _get_index_tensor(grid, +0, +2)
        i_23 = _get_index_tensor(grid, +1, +2)
        i_33 = _get_index_tensor(grid, +2, +2)
          
        p_00 = tf.gather_nd(input_, i_00)
        p_10 = tf.gather_nd(input_, i_10)
        p_20 = tf.gather_nd(input_, i_20)
        p_30 = tf.gather_nd(input_, i_30)
          
        p_01 = tf.gather_nd(input_, i_01)
        p_11 = tf.gather_nd(input_, i_11)
        p_21 = tf.gather_nd(input_, i_21)
        p_31 = tf.gather_nd(input_, i_31)
          
        p_02 = tf.gather_nd(input_, i_02)
        p_12 = tf.gather_nd(input_, i_12)
        p_22 = tf.gather_nd(input_, i_22)
        p_32 = tf.gather_nd(input_, i_32)
          
        p_03 = tf.gather_nd(input_, i_03)
        p_13 = tf.gather_nd(input_, i_13)
        p_23 = tf.gather_nd(input_, i_23)
        p_33 = tf.gather_nd(input_, i_33)
          
        col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
        col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
        col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
        col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
        value = _hermite(col0, col1, col2, col3, y_t)
              
        return value

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        # x_linspace = K_linspace(-1., 1., height)
        # y_linspace = K_linspace(-1., 1., width)
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        transformations = K.reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        new_shape = (batch_size, output_size[0], output_size[1], 2*num_channels)
        f = K.reshape(sampled_grids - regular_grids[:,:2,:], new_shape)
        interpolated_image = self._interpolate(X, f)
        return interpolated_image
