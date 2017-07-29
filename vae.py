import numpy as np
import tensorflow as tf
from collections import OrderedDict


class ccvae():
    def __init__(self, mb=False, len_edge=32, num_channels_10=4, num_channels_20=6, outer=128,
                 middle=256, inner=512, innest=1024, h_dim=1600, latent_dim=900,
                 tb_imgs_to_display=4):
        
        latvisdim = int(np.sqrt(latent_dim))
        self.layer_name_dict = {}

        with tf.name_scope('Input'):
            self.X_10 = tf.placeholder(tf.float32, shape=([None, len_edge, len_edge, num_channels_10]))#bs
            tf.summary.image('input_images', self.X_10[:, :, :, 0:3][:, :, :, ::-1], max_outputs=tb_imgs_to_display)
            
            if mb == True:
                self.X_20 = tf.placeholder(tf.float32, shape=([None, len_edge / 2, len_edge / 2, num_channels_20]))
                tf.summary.image('input_imagess', self.X_20[:, :, :, 0:3], max_outputs=tb_imgs_to_display)

        dw_h_convs = OrderedDict() # dicts for convolutions


        up_h_convs = OrderedDict()
        self.keep_prob = tf.placeholder(tf.float32) # if dropout wanted
        
        ### ENCODING ###

        X_go_10 = self.X_10
        X_go_10 = self.convolute(X_go_10, self.layer_name('conv'), 3, outer, sz = 1)
        X_go_10 = self.convolute(X_go_10, self.layer_name('conv'),3, outer, sz = 1)
        dw_h_convs[1] = self.pooling(X_go_10, 'pool1')
        
        # No pooling of 20 m resolution bands
        if mb == True:
            X_go_20 = self.convolute(self.X_20, self.layer_name('conv20'),3,outer,sz = 1)
            X_go_20 = self.convolute(X_go_20, self.layer_name('conv20'),3,outer,sz = 1)
            dw_h_convs[1] = tf.concat([dw_h_convs[1], X_go_20], 3)


        dw_h_convs[1] = self.convolute(dw_h_convs[1],self.layer_name('conv'),3,middle)
        dw_h_convs[1] = self.convolute(dw_h_convs[1],self.layer_name('conv'),3,middle)
        dw_h_convs[2] = self.pooling(dw_h_convs[1], 'pool2')

        dw_h_convs[2] = self.convolute(dw_h_convs[2],self.layer_name('conv'),3,inner)
        dw_h_convs[2] = self.convolute(dw_h_convs[2],self.layer_name('conv'),3,inner)
        dw_h_convs[3] = self.pooling(dw_h_convs[2], 'pool3')

        dw_h_convs[3] = self.convolute(dw_h_convs[3],self.layer_name('conv'),3,innest)
        dw_h_convs[3] = self.convolute(dw_h_convs[3],self.layer_name('conv'),3,innest)
        dw_h_convs[4] = self.pooling(dw_h_convs[3], 'pool4')

        convdim = 2 * 2 * innest # update    

        ### reparameterization trick

        flattened = tf.reshape(dw_h_convs[4], [-1, convdim])#[4]

        W_enc = tf.get_variable('W_enc', [convdim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_enc = tf.get_variable('b_enc', [h_dim], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
        full1 = tf.nn.dropout(tf.nn.elu(tf.contrib.layers.batch_norm(tf.matmul(flattened, W_enc) + b_enc)), keep_prob=self.keep_prob)

        W_mu = tf.get_variable('W_mu', [h_dim, latent_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_mu = tf.get_variable('b_mu', [latent_dim], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
        mu = tf.matmul(full1, W_mu) + b_mu

        W_logstd = tf.get_variable('W_logstd', [h_dim, latent_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_logstd = tf.get_variable('b_logstd', [latent_dim], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
        logstd = tf.matmul(full1, W_logstd) + b_logstd

        noise = tf.random_normal([1, latent_dim])
        z = mu + tf.multiply(noise, tf.exp(.5*logstd))

        z_visual = tf.reshape(z, [-1, latvisdim, latvisdim, 1])

        tf.summary.image('latents', z_visual, max_outputs=tb_imgs_to_display)

        ### DECODING ###

        W_dec = tf.get_variable('W_dec', [latent_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_dec = tf.get_variable('b_dec', [h_dim], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
        full2 = tf.nn.dropout(tf.nn.elu(tf.contrib.layers.batch_norm(tf.matmul(z, W_dec) + b_dec)), keep_prob=self.keep_prob)

        W_dec2 = tf.get_variable('W_dec2', [h_dim, convdim], initializer=tf.contrib.layers.xavier_initializer())
        b_dec2 = tf.get_variable('b_dec2', [convdim], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
        full3 = tf.nn.elu(tf.contrib.layers.batch_norm(tf.matmul(full2, W_dec2) + b_dec2))

        reshaped = tf.reshape(full3, [-1, 2, 2, innest])

        up_h_convs[0] = tf.image.resize_images(reshaped, [ reshaped.get_shape().as_list()[1]*2, 
                                                                    reshaped.get_shape().as_list()[2]*2])

        up_h_convs[0] = self.convolute(up_h_convs[0], self.layer_name('conv'), 3, innest)
        up_h_convs[0] = self.convolute(up_h_convs[0], self.layer_name('conv'), 3, innest)
        up_h_convs[1] = tf.image.resize_images(up_h_convs[0], [ up_h_convs[0].get_shape().as_list()[1]*2, 
                                                                    up_h_convs[0].get_shape().as_list()[2]*2] ) 

        up_h_convs[1] = self.convolute(up_h_convs[1], self.layer_name('conv'), 3, inner)#[1]
        up_h_convs[1] = self.convolute(up_h_convs[1], self.layer_name('conv'), 3, inner)
        up_h_convs[2] = tf.image.resize_images(up_h_convs[1], [ up_h_convs[1].get_shape().as_list()[1]*2, 
                                                                    up_h_convs[1].get_shape().as_list()[2]*2] ) 

        up_h_convs[2] = self.convolute(up_h_convs[2], self.layer_name('conv'), 3, middle)
        up_h_convs[2] = self.convolute(up_h_convs[2], self.layer_name('conv'), 3, middle)
        up_h_convs[3] = tf.image.resize_images(up_h_convs[2], [ up_h_convs[2].get_shape().as_list()[2]*2, 
                                                                    up_h_convs[2].get_shape().as_list()[2]*2] )

        # split 20 m resolution data:

        if mb == True:
            up_h_convs[3] = self.convolute(up_h_convs[3], self.layer_name('conv'), 3, outer * 2)
            up_h_convs[3] = self.convolute(up_h_convs[3], self.layer_name('conv'), 3, outer * 2)
            up_h_convs[3] = self.tf.slice(up_h_convs[3], [0, 0, 0, 0], [bs, len_edge / 2, len_edge / 2, outer])
            lowres = tf.slice(up_h_convs[3], [0, 0, 0, outer], [bs, len_edge / 2, len_edge / 2, outer])
            #conv and reconstruct lowres
            lowres = self.convolute(lowres, layer_name('conv'), 3, outer)
            W_low = tf.get_variable('weights_low', [1, 1, outer, num_channels_20], initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=False)#, name='weights')
            b_low = tf.get_variable('biases_low', [num_channels_20], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
            self.reconstruction_low = tf.nn.sigmoid(tf.nn.conv2d(lowres, W_low, strides=[1, 1, 1, 1], padding='SAME') + b_low)

        up_h_convs[3] = self.convolute(up_h_convs[3], self.layer_name('conv'), 3, outer)
        #up_h_convs[4] = tf.image.resize_images(up_h_convs[3], [ up_h_convs[3].get_shape().as_list()[2]*2, 
        #                                                            up_h_convs[3].get_shape().as_list()[2]*2] )
        #up_h_convs[4] = convolute(up_h_convs[4], layer_name('conv'), 3, outer)
        #highres = convolute(highres, layer_name('conv'), 3, outer)
        W_rec = tf.get_variable('weights_rec', [1, 1, outer, 4], initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=False)#, name='weights')
        b_rec = tf.get_variable('biases_rec', [4], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
        self.reconstruction = tf.nn.sigmoid(tf.nn.conv2d(up_h_convs[3], W_rec, strides=[1, 1, 1, 1], padding='SAME') + b_rec)


        ### VISUALIZE OUTPUT

        with tf.name_scope('reconst'):
            tf.summary.image('reconstructed_images', self.reconstruction[:, :, :, 0:3][:, :, :, ::-1], max_outputs=tb_imgs_to_display)
            if mb == True:
                tf.summary.image('reconstructed_imagess', self.reconstruction_low[:, :, :, 0:3], max_outputs=tb_imgs_to_display)


        ### CALCULATE LOSSES

        X_flat = tf.contrib.layers.flatten(self.X_10)
        R_flat = tf.contrib.layers.flatten(self.reconstruction)
        self.log_likelihood = tf.reduce_sum(X_flat*tf.log(R_flat + 1e-9)+(1 - X_flat)*tf.log(1 - R_flat + 1e-9), reduction_indices=1)

        if mb == True:
            X2_flat = tf.contrib.layers.flatten(X_20)
            R2_flat = tf.contrib.layers.flatten(reconstruction_low)
            log_likelihood2 = tf.reduce_sum(X2_flat*tf.log(R2_flat + 1e-9)+(1 - X2_flat)*tf.log(1 - R2_flat + 1e-9), reduction_indices=1)
            self.log_likelihood = self.log_likelihood + log_likelihood2

        #log_likelihood = tf.reduce_sum(X_flat*tf.log(R_flat + 1e-9)+(1 - X_flat)*tf.log(1 - R_flat + 1e-9), reduction_indices=1)
        tf.summary.scalar('LogLike', tf.reduce_mean(self.log_likelihood))

        self.KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)
        tf.summary.scalar('KL', tf.reduce_mean(self.KL_term))

        self.variational_lower_bound = tf.reduce_mean(self.log_likelihood - self.KL_term)
        tf.summary.scalar('cost', self.variational_lower_bound)

        self.validator = tf.cast(self.variational_lower_bound, tf.int32) # alibi
        
    # HELPER FUNCTIONS
        
    def layer_name(self, base_name):
        if base_name not in self.layer_name_dict:
            self.layer_name_dict[base_name] = 0
        self.layer_name_dict[base_name] += 1
        name = base_name + str(self.layer_name_dict[base_name])
        return name

    def convolute(self, inp, name, kernel_size = 3, out_chans = 64, sz = 1):
        inp_chans = inp.get_shape().as_list()[-1]
        with tf.name_scope(name):
            with tf.variable_scope(name) as scope:
                W = tf.get_variable('weights', [kernel_size, kernel_size, inp_chans, out_chans], initializer=tf.contrib.layers.xavier_initializer_conv2d())#, regularizer=tf.contrib.layers.l2_regularizer(0.0005))#, name='weights')
                b = tf.get_variable('biases', [out_chans], initializer=tf.constant_initializer(0.1), regularizer=None, dtype=tf.float32)
                conv = tf.nn.conv2d(inp, W, strides=[1, sz, sz, 1], padding='SAME')
                conv = tf.contrib.layers.batch_norm(conv, scope=scope) # train?
                conv = tf.nn.elu(conv+b)
                tf.summary.histogram('weights', W)
                tf.summary.histogram('biases', b)
                tf.summary.histogram('activations', conv)
    #        conv = tf.nn.dropout(conv, 0.8)
        return conv

    def pooling(self, inp, name, factor=2):
        pool = tf.nn.max_pool(inp, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
        return pool