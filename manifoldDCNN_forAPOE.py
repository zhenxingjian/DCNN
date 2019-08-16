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


lr = 0.1
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
    group_test_times = 3000

    ################

    _,_,Label = read_data(label_CSV,label_name,recalculate = False)


    NagPos = np.where(Label)
    PosPos = np.where(1-Label)
    pdb.set_trace()

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
    NagData = np.concatenate((NagData,NagData,NagData,NagData),axis = 0)
    PosData = np.concatenate((PosData,PosData,PosData,PosData),axis = 0)
    Data = np.concatenate((NagData,PosData),axis = 0)

    Nag_lastvoxel = NagData[:,[-1],:,:]
    Pos_lastvoxel = PosData[:,[-1],:,:]

    NagData = NagData[:,:-1,:,:]
    PosData = PosData[:,:-1,:,:]

    pdb.set_trace()

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
    # with tf.Session() as sess:
        ########## pre-train with all data
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ##########
        dW0_ = []
        for k__ in range(1):
            saver.restore(sess, "./data/ad_data/processed_data/"+label_name+"/model/model"+str(Trackids[0])+".ckpt")
            # pdb.set_trace()
            starttime = time.time()
            for epoch in range(epoch_num):
                _, loss_ = sess.run([train_step,loss],
                                            feed_dict={
                                                       X:NagData,
                                                       Y_pred_voxel:Nag_lastvoxel,
                                                       keep_prob:0.75
                                                        })
            Weights_Mani_Nag,W_FM_Nag = sess.run([Weights_Mani,W_FM])
            # print (loss_)
            saver.restore(sess, "./data/ad_data/processed_data/"+label_name+"/model/model"+str(Trackids[0])+".ckpt")
            for epoch in range(epoch_num):
                _, loss_ = sess.run([train_step,loss],
                                            feed_dict={
                                                       X:PosData,
                                                       Y_pred_voxel:Pos_lastvoxel,
                                                       keep_prob:0.75
                                                        })
            Weights_Mani_Pos,W_FM_Pos = sess.run([Weights_Mani,W_FM])
            print (time.time()-starttime)
            dW0 = 0.
            for i in range(depth):
                w1 = np.reshape(np.square(Weights_Mani_Nag[i]["W_DCNN1_root"]),[-1,middle_channels[i]])
                w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
                w2 = np.reshape(np.square(Weights_Mani_Pos[i]["W_DCNN1_root"]),[-1,middle_channels[i]])
                w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))

                dW0 = dW0 + np.mean(np.square(w1-w2)) 
                w1 = np.reshape(np.square(Weights_Mani_Nag[i]["W_DCNN2_root"]),[-1,out_channels[i]])
                w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
                w2 = np.reshape(np.square(Weights_Mani_Pos[i]["W_DCNN2_root"]),[-1,out_channels[i]])
                w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))
                dW0 = dW0 + np.mean(np.square(w1-w2))
                w1 = (np.square(Weights_Mani_Nag[i]["W_res_wFW_root"]))
                w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
                w2 = (np.square(Weights_Mani_Pos[i]["W_res_wFW_root"]))
                w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))
                dW0 = dW0 + np.mean(np.square(w1-w2))

            w1 = W_FM_Nag
            w2 = W_FM_Pos
            dW0 = dW0 + np.mean(np.square(w1-w2))
            dW0_.append(dW0)
        dW0_ = np.asarray(dW0_)
        dW0 = np.mean(dW0_)
        print("Original distance is: ",dW0)
        print (dW0_)

        count = 0.
        for rep in range(group_test_times):
            POS_ = range(class1_num+class2_num)
            random.shuffle(POS_)
            FakeNagPos = POS_[0:class1_num]
            FakePosPos = POS_[class1_num:]

            FakeNagData = np.reshape(Data[FakeNagPos],[-1,matrix_length,n_para,in_channels[0]])
            FakePosData = np.reshape(Data[FakePosPos],[-1,matrix_length,n_para,in_channels[0]])

            FakeNag_lastvoxel = FakeNagData[:,[-1],:,:]
            FakePos_lastvoxel = FakePosData[:,[-1],:,:]

            FakeNagData = FakeNagData[:,:-1,:,:]
            FakePosData = FakePosData[:,:-1,:,:]

            saver.restore(sess, "./data/ad_data/processed_data/"+label_name+"/model/model"+str(Trackids[0])+".ckpt")

            for epoch in range(epoch_num):
                _, loss_ = sess.run([train_step,loss],
                                            feed_dict={
                                                       X:FakeNagData,
                                                       Y_pred_voxel:FakeNag_lastvoxel,
                                                       keep_prob:0.75
                                                        })
            Weights_Mani_Nag,W_FM_Nag = sess.run([Weights_Mani,W_FM])
            # print (loss_)
            saver.restore(sess, "./data/ad_data/processed_data/"+label_name+"/model/model"+str(Trackids[0])+".ckpt")
            for epoch in range(epoch_num):
                _, loss_ = sess.run([train_step,loss],
                                            feed_dict={
                                                       X:FakePosData,
                                                       Y_pred_voxel:FakePos_lastvoxel,
                                                       keep_prob:0.75
                                                        })
            Weights_Mani_Pos,W_FM_Pos = sess.run([Weights_Mani,W_FM])
            # print (loss_)

            dW = 0
            for i in range(depth):
                w1 = np.reshape(np.square(Weights_Mani_Nag[i]["W_DCNN1_root"]),[-1,middle_channels[i]])
                w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
                w2 = np.reshape(np.square(Weights_Mani_Pos[i]["W_DCNN1_root"]),[-1,middle_channels[i]])
                w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))
                dW = dW + np.mean(np.square(w1-w2)) 
                w1 = np.reshape(np.square(Weights_Mani_Nag[i]["W_DCNN2_root"]),[-1,out_channels[i]])
                w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
                w2 = np.reshape(np.square(Weights_Mani_Pos[i]["W_DCNN2_root"]),[-1,out_channels[i]])
                w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))
                dW = dW + np.mean(np.square(w1-w2))
                w1 = (np.square(Weights_Mani_Nag[i]["W_res_wFW_root"]))
                w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
                w2 = (np.square(Weights_Mani_Pos[i]["W_res_wFW_root"]))
                w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))
                dW = dW + np.mean(np.square(w1-w2))
            
            # w1 = (np.square(W_FM_Nag))
            # w1 = np.divide(w1,np.sum(w1,axis = 0,keepdims = True))
            # w2 = (np.square(W_FM_Pos))
            # w2 = np.divide(w2,np.sum(w2,axis = 0,keepdims = True))
            w1 = W_FM_Nag
            w2 = W_FM_Pos
            dW = dW + np.mean(np.square(w1-w2))
            print("Fake distance ", rep,  " is: ",dW, "And small is :",dW < dW0 )
            if dW < dW0:
                count = count + 1.
            print("Rate is : ",count/(rep+1))

        print (Trackids,"   The rate of random is smaller than the original is : " , count/group_test_times)
        np.savez("./data/ad_data/processed_data/"+label_name+"/Result"+str(Trackids[0])+".npz",rate = np.asarray(count/group_test_times),In_channel = np.asarray(in_channels),Middle_channel = np.asarray(middle_channels), Out_channel = np.asarray(out_channels),K = np.asarray(k))
