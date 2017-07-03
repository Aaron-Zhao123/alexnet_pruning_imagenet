import tensorflow as tf
import numpy as np
import pickle
import sys
from tensorflow.contrib.layers.python.layers import layers as layers_lib

"""
gives back self.pred, self.
"""
class alexnet(object):
    def __init__(self, isLoad):
        self._get_variables(isLoad)
        self._init_weight_masks(isLoad)
        self.isLoad = isLoad
        # self.conv_network()

    def loss(logits, labels):
        """for constructing the graph"""
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logtis, labels, name = 'cross_entropy_batchwise')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
        """
        puts cross entropy as a loss into collection
        potentially, there can be L1 and L2 losses added to this "losses" term
        in collections
        """
        tf.add_to_collection('losses', cross_entropy_mean)


    def error_rates(self,topk = 1):
        """
        Args:
            self.pred: shape [B,C].
            self.labels: shape [B].
            topk(int): topk
        Returns:
            a float32 vector of length N with 0/1 values. 1 means incorrect
            prediction.
        """
        return tf.cast(tf.logical_not(tf.nn.in_top_k(self.pred, self.labels, topk)),
            tf.float32)

    def conv_network(self, images, keep_prob):
        self.keep_prob = keep_prob
        imgs = images

        conv1 = self.conv_layer(imgs, 'conv1', padding = 'SAME', stride = 4, prune = True)
        pool1 = self.maxpool(conv1, 'pool1', 3, 2, padding = 'VALID')
        lrn1 = self.lrn(pool1, 'lrn1')
        print(pool1)

        conv2 = self.conv_layer(lrn1, 'conv2', prune = True, split = 2)
        pool2 = self.maxpool(conv2, 'pool2', 3, 2, padding = 'VALID')
        lrn2 = self.lrn(pool2, 'lrn2')
        # norm2 = self.batch_norm(conv2, 'norm2', train_phase = self.isTrain)
        print(pool2)

        conv3 = self.conv_layer(lrn2, 'conv3', prune = True)
        # norm3 = self.batch_norm(conv3, 'norm3', train_phase = self.isTrain)
        # pool3 = self.maxpool(norm3, 'pool3', 3, 2)
        conv4 = self.conv_layer(conv3, 'conv4', prune = True, split = 2)
        # norm4 = self.batch_norm(conv4, 'norm4', train_phase = self.isTrain)
        conv5 = self.conv_layer(conv4, 'conv5', prune = True, split = 2)
        # norm5 = self.batch_norm(conv5, 'norm5', train_phase = self.isTrain)
        pool5 = self.maxpool(conv5, 'pool5', 3, 2, padding = 'VALID')
        # sys.exit()
        print(pool5)


        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = self.fc_layer(flattened, 'fc6', prune = True)
        fc6_drop = self.dropout_layer(fc6, 'fc6')
        # norm6 = self.batch_norm(fc6, 'norm6', train_phase = self.isTrain)

        fc7 = self.fc_layer(flattened, 'fc7', prune = True)
        fc7_drop = self.dropout_layer(fc7, 'fc7')
        # norm7 = self.batch_norm(fc7, 'norm7', train_phase = self.isTrain)

        fc8 = self.fc_layer(fc7_drop, 'fc8', prune = True, apply_relu = False)
        self.pred = fc8
        return self.pred

    def maxpool(self, x, name, filter_size, stride, padding = 'SAME'):
        return tf.nn.max_pool(x, ksize = [1, filter_size, filter_size, 1],
            strides = [1, stride, stride, 1], padding = padding, name = name)

    def lrn(self, x, name, depth_radius = 2, bias = 1.0, alpha = 2e-5, beta = 0.75):
        """
        local response normalization
        ref: https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
        """
        return tf.nn.lrn(x, depth_radius = depth_radius, bias = bias,
            alpha = alpha, beta = beta, name = name)

    def batch_norm(self, x, name, train_phase, data_format = 'NHWC', epsilon = 1e-3):
        """
        TODO: this batch norm has an error
        refs:
        1. https://github.com/ppwwyyxx/tensorpack/blob/a3674b47bfbf0c8b04aaa85d428b109fea0128ca/tensorpack/models/batch_norm.py
        2. https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
        """
        shape = x.get_shape().as_list()
        ndims = len(shape)

        assert ndims in [2,4]
        if ndims == 2:
            data_format = 'NHWC'

        if data_format == 'NCHW':
            n_out = shape[1]
        else:
            n_out = shape[-1]  # channel

        assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"

        with tf.variable_scope(name):
            beta = tf.Variable(tf.constant(0.0, shape = [n_out]),
                name = 'beta', trainable = True)
            gamma = tf.Variable(tf.constant(1.0, shape = [n_out]),
                name = 'gamma', trainable = True)
            axis = list(range(len(x.get_shape())-1))
            batch_mean, batch_var = tf.nn.moments(x, axis, name = 'moments')

            ema = tf.train.ExponentialMovingAverage(decay = 0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(train_phase,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
        return normed

    def dropout_layer(self, x, name):
        return layers_lib.dropout(
          x, self.keep_prob, is_training= (not self.isLoad), scope=name)
        # return tf.nn.dropout(x, self.keep_prob)

    def fc_layer(self, x, name, prune = False, apply_relu = True):
        with tf.variable_scope(name, reuse = True):
            with tf.device('/cpu:0'):
                w = tf.get_variable('w')
                b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            ret = tf.nn.xw_plus_b(x,w,b)
            if apply_relu:
                ret = tf.nn.relu(ret)
        return ret

    def conv_layer(self, x, name, padding = 'SAME', stride = 1,
        split = 1, data_format = 'NHWC', prune = False):

        channel_axis = 3 if data_format == 'NHWC' else 1
        with tf.variable_scope(name, reuse = True):
            with tf.device('/cpu:0'):
                w = tf.get_variable('w')
                b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            if split == 1:
                conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding, data_format=data_format)
                # conv = tf.nn.conv2d(x, w, stride, padding)
            else:
                inputs = tf.split(x, split, channel_axis)
                kernels = tf.split(w, split, 3)
                # outputs = [tf.nn.conv2d(i, k, stride, padding)
                outputs = [tf.nn.conv2d(i, k, [1, stride, stride, 1], padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            # using Relu
            ret = tf.nn.relu(tf.nn.bias_add(conv, b, data_format=data_format), name='output')
        return ret

    def _get_variables(self, isload, weights_path = 'DEFAULT'):
        """
        Network architecture definition
        """
        self.keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        kernel_shapes = [
            [11, 11, 3, 96],
            [5, 5, 48, 256],
            [3, 3, 256, 384],
            [3, 3, 192, 384],
            [3, 3, 192, 256],
            [6 * 6 * 256, 4096],
            [4096, 4096],
            [4096, 1001]
        ]
        biase_shape = [
            [96],
            [256],
            [384],
            [384],
            [256],
            [4096],
            [4096],
            [1001]
        ]
        self.weight_shapes = kernel_shapes
        self.biase_shapes = biase_shape
        if isload:
            with open(weights_path+'.npy', 'rb') as f:
                weights, biases = pickle.load(f)
            for i, key in enumerate(self.keys):
                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key,
                    w_init = weights[key],
                    b_init = biases[key])
        else:
            for i,key in enumerate(self.keys):
                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key)

    def _init_layerwise_variables(self, w_shape, b_shape, name, w_init = None, b_init = None):
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                if w_init is None:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
                else:
                    w_init = tf.constant(w_init)
                if b_init is None:
                    b_init = tf.constant_initializer()
                else:
                    b_init = tf.constant(b_init)
                w = tf.get_variable('w', w_shape, initializer = w_init)
                b = tf.get_variable('b', b_shape, initializer = b_init)

    def _init_weight_masks(self, is_load):
        names = self.keys
        if is_load:
            with open(weights_path+'mask.npy', 'rb') as f:
                self.weights_masks, self.biases_masks= pickle.load(f)
        else:
            self.weights_masks = {}
            self.biases_masks = {}
            for i, key in enumerate(names):
                self.weights_masks[key] = np.ones(self.weight_shapes[i])
                self.biases_masks[key] = np.ones(self.biase_shapes[i])

    def _apply_a_mask(self, mask, var):
        return (var * mask)
