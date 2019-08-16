from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import pdb
import math
import time
import os
import sys

from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences

from readdata import read_data, load_length
from matrixcell import CNNRNNCell
from manifoldDCNN import ManiDist, res_wFM, DCNN, res_block, last_layer, last_layer_multi, last_layer_mean

lr = 0.9
decay_steps = 1000
decay_rate = 0.99

global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)

DataName = "DTI"

if DataName == "DTI":

    mode = "SPD"

    in_channels = [1,3,5]
    middle_channels = [3,3,8]
    out_channels = [3,5,10]

    matrix_size = 3
    n = matrix_size
    n_para = n*n

    pre_train_epoch = 5000
    epoch_num = 500 #50
    depth = len(out_channels)
    assert len(in_channels) == len (middle_channels) and len (middle_channels) == len(out_channels)

    k = 3
    d0 = 1
    ############# pre-process data part
    label_CSV = "data/ad_data/APOE.csv"
    label_name = "APOE"
    class_num = 2 ########
    group_test_times = 5000
    ################

    _,_,Label = read_data(label_CSV,label_name,recalculate = False)

    NagPos = np.where(Label)
    PosPos = np.where(1-Label)

    Length, Nampyte = load_length("./data/ad_data/processed_data/"+label_name+"/Track_info.txt")

    Trackids = [int(sys.argv[1])] ###### all fibers
    Track_num = len(Trackids)

    matrix_length_all = np.zeros(Track_num, dtype = np.int32)
    for i in range(Track_num):
        matrix_length_all[i] = Length[Trackids[i]][1]

    matrix_length = np.sum(matrix_length_all)

    print("In  channels is : ",in_channels)
    print("Mid channels is : ", middle_channels)
    print("Out Channels is : ",out_channels)
    print("k            is : ",k)
    print("fiber number is : ",Trackids)
    print("fiber length is : ",matrix_length)

    data = []
    label = []

    Data = None
    for i in range(Track_num):
        if Data is None:
            Data = np.load("./data/ad_data/processed_data/"+label_name+"/spdTrack"+str(Trackids[i])+".npy")
        else:
            tempdata = np.load("./data/ad_data/processed_data/"+label_name+"/spdTrack"+str(Trackids[i])+".npy")
            Data = np.concatenate((Data,tempdata),axis = 1 )


    class1_num = NagPos[0].size
    class2_num = PosPos[0].size
    # pdb.set_trace()
    NagData = np.reshape(Data[NagPos],[-1,matrix_length,n_para,in_channels[0]])
    PosData = np.reshape(Data[PosPos],[-1,matrix_length,n_para,in_channels[0]])

    Data = np.concatenate((NagData,PosData),axis = 0)

    Nag_lastvoxel = NagData[:,[-1],:,:]
    Pos_lastvoxel = PosData[:,[-1],:,:]

    NagData = NagData[:,:-1,:,:]
    PosData = PosData[:,:-1,:,:]

    X = tf.placeholder(np.float32,shape = (None,matrix_length-1,n_para,in_channels[0])) 
    Y_pred_voxel = tf.placeholder(np.float32,shape = (None,1,n_para,in_channels[0]))

    keep_prob = tf.placeholder(tf.float32)


    Weights_Mani = []

    for i in range(depth):
        Weights_Mani.append({
            'W_DCNN1_root':tf.Variable(tf.random_uniform([k,in_channels[i],middle_channels[i]],minval = 0.01, maxval = 0.99)),
            'W_DCNN2_root':tf.Variable(tf.random_uniform([k,middle_channels[i],out_channels[i]],minval = 0.01, maxval = 0.99)),
            'W_res_wFW_root':tf.Variable(tf.random_uniform([in_channels[i]+out_channels[i],out_channels[i]],minval = 0.01, maxval = 0.99)),
          })


    W_FM = tf.Variable(tf.random_uniform([out_channels[i],1],minval = 0.01, maxval = 0.99))


    tf.keras.backend.set_learning_phase(True)

    layer_ = X

    for i in range(depth):
        layer_ = res_block(layer_,d0*(2**i),Weights_Mani[i],mode)

    pred_last_voxel = last_layer_mean(layer_,W_FM)
    total_size = tf.shape(X)[0]
    pred_last_voxel = tf.reshape(pred_last_voxel,[total_size,1,n_para,in_channels[0]])

    loss = tf.reduce_mean(tf.pow(pred_last_voxel-Y_pred_voxel,2))


    with tf.control_dependencies([add_global]):
        opt = tf.train.AdadeltaOptimizer(learning_rate)
        train_step = opt.minimize(loss)

    ################# Group analysis
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(pre_train_epoch):
            _, loss_ = sess.run([train_step,loss],
                                        feed_dict={
                                                   X:Data[:,:-1,:,:],
                                                   Y_pred_voxel:Data[:,[-1],:,:],
                                                   keep_prob:0.75
                                                    })
        saver = tf.train.Saver()
        if not os.path.isdir("./data/ad_data/processed_data/"+label_name+"/model/"):
            os.mkdir("./data/ad_data/processed_data/"+label_name+"/model/")
        saver.save(sess, "./data/ad_data/processed_data/"+label_name+"/model/model"+str(Trackids[0])+".ckpt")
        ##########
os.system("python manifoldDCNN_forAPOE.py "+sys.argv[1])