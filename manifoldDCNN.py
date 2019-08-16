from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import pdb
import math
import time
import os

from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences

from readdata import read_data, load_length
from matrixcell import CNNRNNCell

def ManiDist(X1,X2,mode):
    '''
    X1 with shape batch * n_para
    X2 with shape batch * n_para
    No Sequence. Every point is just a batch of data point, with different number of channels.
    '''
    if mode == "SPD":
        batch_size = tf.shape(X1)[0]#X1.shape[0]
        n_para = np.int(X1.shape[1])
        if n_para < 1:
            MatrixN = np.sqrt(n_para).astype(np.int32)
            X1 = tf.reshape(X1,[batch_size , MatrixN , MatrixN])
            X1 = (X1+tf.transpose(X1,[0,2,1]))/2
            X2 = tf.reshape(X2,[batch_size , MatrixN , MatrixN])
            X2 = (X2+tf.transpose(X2,[0,2,1]))/2
            Dist = tf.log(tf.linalg.det((X1+X2)/2))-0.5*tf.log(tf.linalg.det(tf.matmul(X1,X2)))
            Dist = tf.reshape(Dist,[batch_size,1])
        else:
            squareDist = tf.reduce_sum(tf.pow(X1-X2,2),axis = 1)
            Dist = tf.sqrt(squareDist)
            Dist = tf.reshape(Dist,[batch_size,1])
        return Dist
    elif mode == "ODF":
        # Not using this one
        # Because can do it in matrix version
        return None

def res_wFM(dx,x,W_root,mode):
    '''
    The residual block
    dx is the output of two DCNN
    x is the origianl input x
    W_root shape is [in_channel+out_channel,out_channel]
    mode is "SPD" or "ODF"
    '''
    W = tf.pow(W_root,2)
    W_sum = tf.reduce_sum(W,0)
    W = tf.div(W,W_sum)
    batch_size = tf.shape(dx)[0]#dx.shape[0]
    sequence_length = dx.shape[1]
    n_para = dx.shape[2]
    out_channel = dx.shape[3]
    in_channel = x.shape[3]
    if mode == "SPD":
        x_dx = tf.concat([x,dx],axis = 3)
        x_dx = tf.reshape(x_dx,[batch_size*sequence_length*n_para,(in_channel+out_channel)])
        x_dx = tf.matmul(x_dx,W)
        x_dx = tf.reshape(x_dx,[batch_size,sequence_length,n_para,out_channel])
        return x_dx

    elif mode == "ODF":
        x_dx = tf.concat([x,dx],axis = 3)
        x_dx = tf.reshape(x_dx,[batch_size*sequence_length*n_para,(in_channel+out_channel)])
        x_dx = tf.matmul(x_dx,W)
        x_dx = tf.reshape(x_dx,[batch_size,sequence_length,n_para,out_channel])
        return x_dx

def DCNN(x,d,W_root,mode):
    ''' 
    x is input, with shape batch * sequence_length * n_para * in_channel
    d is the number of skipped, a number
    w is the weights, with shape k * in_channel * out_channel
    mode is "SPD" or "ODF"
    '''
    W = tf.pow(W_root,2)

    batch_size = tf.shape(x)[0]#x.shape[0]
    sequence_length = x.shape[1]
    n_para = x.shape[2]
    k = W.shape[0]
    in_channel = W.shape[1]
    out_channel = W.shape[2]

    padding = (k - 1) * d
    x_pad = tf.pad(x,tf.constant([(0,0),(1,0),(0,0),(0,0)]) * padding , "REFLECT") # for the first element, we need padding
    W = tf.reshape(W,[k*in_channel,out_channel])
    W_sum = tf.reduce_sum(W,0)
    W = tf.div(W,W_sum) # constrain sum(w_k_inchannel) = 1
    if mode =="SPD":
        x_reorder = tf.transpose(x_pad,[0,2,1,3])
        x_reshape = tf.reshape(x_reorder,[batch_size*n_para,1,sequence_length+padding,in_channel])

        W = tf.reshape(W,[1,k,in_channel,out_channel])
        conv1 = tf.nn.atrous_conv2d(x_reshape,W,d,"VALID",name=None)
        conv1 = tf.reshape(conv1,[batch_size,n_para,sequence_length,out_channel])
        conv1 = tf.transpose(conv1,[0,2,1,3])
        return conv1
        
    elif mode == "ODF":
        x_reorder = tf.transpose(x_pad,[0,2,1,3])
        x_reshape = tf.reshape(x_reorder,[batch_size*n_para,1,sequence_length+padding,in_channel])

        W = tf.reshape(W,[1,k,in_channel,out_channel])
        conv1 = tf.nn.atrous_conv2d(x_reshape,W,d,"VALID",name=None)
        conv1 = tf.reshape(conv1,[batch_size,n_para,sequence_length,out_channel])
        conv1 = tf.transpose(conv1,[0,2,1,3])
        return conv1

def res_block(x,d,W,mode):
    '''
    x is input
    d is the number of skipped, a number
    W is a list of length 3. 
    W[0] and W[1] is the Weights for DCNN
    W[2] is the Weights for res_wFM
    mode is "SPD" or "ODF"
    '''
    Y_layer1 = DCNN(x,d,W["W_DCNN1_root"],mode)
    Y_layer2 = DCNN(Y_layer1,d,W["W_DCNN2_root"],mode)
    Y_out = res_wFM(Y_layer2,x,W["W_res_wFW_root"],mode)
    # return Y_layer2
    return Y_out

def last_layer(x,mode):
    '''
    x with shape batch_size x sequence_length x n_para x channels

    And this is before Softmax!
    '''
    batch_size = tf.shape(x)[0]#np.int32(x.shape[0])
    sequence_length = np.int32(x.shape[1])
    n_para = np.int32(x.shape[2])
    channels = np.int32(x.shape[3])

    X_slice = tf.slice(x,[0,sequence_length-1,0,0],[batch_size,1,n_para,channels]) # shape batch_size x 1 x 9 x channels
    X_slice = tf.reshape(X_slice,[batch_size,n_para,channels])
    if mode == "SPD":
        M_mean = tf.reduce_mean(X_slice,axis = 2) # shape batch_size x 16x16
        oi = None
        for channel_idx in range(channels):
            temp_X = tf.slice(X_slice,[0,0,channel_idx],[-1,-1,1])
            temp_X = tf.reshape(temp_X,[batch_size,n_para])
            if oi is None:
                oi = ManiDist(temp_X,M_mean,mode)
            else:
                oi = tf.concat([oi,ManiDist(temp_X,M_mean,mode)],axis = 1)

        # oi shape batch_size x channels
        

        return oi
    elif mode == "ODF":
        M_mean = tf.reduce_mean(X_slice,axis = 2)
        M_mean_sum = tf.reduce_sum( tf.pow(M_mean,2), axis = 1 , keepdims = True )
        M_mean = tf.div(M_mean,M_mean_sum)

        X_reorder = tf.transpose(X_slice,[0,2,1])
        oi = tf.matmul(X_reorder,tf.reshape(M_mean,[batch_size,n_para,1]))

        oi = tf.reshape(oi,[batch_size,channels])
        # oi shape batch_size x channels

        return oi

def last_layer_multi(x,W,mode):
    '''
    x with shape batch_size x sequence_length x n_para x channels
    W with shape channels x num_clusters
    And this is before Softmax!
    '''
    batch_size = tf.shape(x)[0]#np.int32(x.shape[0])
    sequence_length = np.int32(x.shape[1])
    n_para = np.int32(x.shape[2])
    channels = np.int32(x.shape[3])
    n_cluster = np.int32(W.shape[1])

    W = tf.pow(W,2)
    W_sum = tf.reduce_sum(W,0)
    W = tf.div(W,W_sum)

    X_slice = tf.slice(x,[0,sequence_length-1,0,0],[batch_size,1,n_para,channels]) # shape batch_size x 1 x 9 x channels
    X_slice = tf.reshape(X_slice,[batch_size,n_para,channels])
    if mode == "SPD":
        M_mean = tf.matmul(tf.reshape(X_slice,[batch_size*n_para,channels]),W)
        M_mean = tf.reshape(M_mean,[batch_size,n_para,n_cluster])
        # pdb.set_trace()
        oi = None
        for channel_idx in range(channels):
            temp_X = tf.slice(X_slice,[0,0,channel_idx],[-1,-1,1])
            temp_X = tf.reshape(temp_X,[batch_size,n_para])
            temp_oi = None
            for cluster_idx in range(n_cluster):
                temp_mean = tf.slice(M_mean,[0,0,cluster_idx],[-1,-1,1])
                temp_mean = tf.reshape(temp_mean,[batch_size,n_para])
                if temp_oi is None:
                    temp_oi = ManiDist(temp_X,temp_mean,mode)
                else:
                    temp_oi = tf.concat([temp_oi,ManiDist(temp_X,temp_mean,mode)],axis = 1)

            if oi is None:
                oi = tf.reshape(temp_oi,[batch_size,n_cluster,1])
            else:
                oi = tf.concat([oi,tf.reshape(temp_oi,[batch_size,n_cluster,1])],axis = 2)
        oi = tf.reshape(oi,[batch_size,n_cluster,channels])
        oi = tf.reduce_max(oi,axis = 1)
        # oi shape batch_size x (channels x clusters)
        
        return oi
    elif mode == "ODF":
        M_mean = tf.reduce_mean(X_slice,axis = 2)
        M_mean_sum = tf.reduce_sum( tf.pow(M_mean,2), axis = 1 , keepdims = True )
        M_mean = tf.div(M_mean,M_mean_sum)
        # pdb.set_trace()
        X_reorder = tf.transpose(X_slice,[0,2,1])
        oi = tf.matmul(X_reorder,tf.reshape(M_mean,[batch_size,n_para,1]))

        oi = tf.reshape(oi,[batch_size,channels])
        # oi shape batch_size x channels

        return oi

def last_layer_mean(x,W):
    '''
    x with shape batch_size x sequence_length x n_para x channels
    W with shape channels x num_clusters
    And this is before Softmax!
    '''
    batch_size = tf.shape(x)[0]#np.int32(x.shape[0])
    sequence_length = np.int32(x.shape[1])
    n_para = np.int32(x.shape[2])
    channels = np.int32(x.shape[3])

    X_slice = tf.slice(x,[0,sequence_length-1,0,0],[batch_size,1,n_para,channels]) # shape batch_size x 1 x 9 x channels
    X_slice = tf.reshape(X_slice,[batch_size,n_para,channels])
    M_mean = tf.matmul(tf.reshape(X_slice,[batch_size*n_para,channels]),W)
    M_mean = tf.reshape(M_mean,[batch_size,n_para])

    return M_mean
