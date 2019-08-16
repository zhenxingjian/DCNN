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
from manifoldDCNN import ManiDist, res_wFM, DCNN, res_block, last_layer, last_layer_multi, last_layer_mean

lr = 0.8
decay_steps = 1000
decay_rate = 0.99

global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)

DataName = "UCF11"

if DataName == "UCF11":
    mode = "SPD"
    def readUCFdata(idx,datapath,matrix_length,k):
        '''
        k is every k samples take one.
        for example, a = [1,2,3,4,5,6,7]; k = 2; result is [1,3,5,7]
        '''
        datas = np.load(datapath+'data'+str(idx)+'.npy')
        labels = np.load(datapath+'label'+str(idx)+'.npy')
        lengths = np.load(datapath+'length'+str(idx)+'.npy')
        data = []
        for i in range(len(datas)):
            # print i
            datas[i] = datas[i][range(0,lengths[i],k),...]
            # pdb.set_trace()
            data.append(  pad_sequences([datas[i]], maxlen=matrix_length, truncating='post', dtype='float32')[0] )
        data = np.asarray(data)
        return data,labels

    def Chol_de(A,n,batch_size):
        '''
        input matrix A and it's size n
        decomponent by Cholesky
        return a vector with size n*(n+1)/2
        '''
        #A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
        # A = tf.cond( 
        #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) , 
        #     lambda: A, 
        #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
        #L = tf.cholesky(A)

        L = A
        result = tf.slice(L,[0,0,0],[-1,1,1])
        for i in range(1,n):
            j = i
            result = tf.concat( [result , tf.slice(L,[0,i,0],[-1,1,j+1])],axis = 2 )

        result = tf.reshape(result,[-1,n*(n+1)//2])
        return result

    batch_size = 40
    batch_num = 40
    height = 120
    width = 160
    in_channel = 3
    out_channel = 6
    tot_time_points = 50
    matrix_length = tot_time_points
    class_num = 11
    matrix_size = out_channel+1
    n = matrix_size
    epoch_num = 1001
    sample_rate = 3

    in_channels = [1,3,4]
    middle_channels = [3,3,4]
    out_channels = [3,4,4]

    depth = len(in_channels)
    assert len(in_channels) == len (middle_channels) and len (middle_channels) == len(out_channels)
    k = 5
    d0 = 1

    CNN_kernel_shape = [[7,7,4],[7,7,out_channel]]
    CNN_num_layer = len(CNN_kernel_shape)

    reduced_spatial_dim = height * width / (4**CNN_num_layer) 
    beta = 0.3

    X = tf.placeholder(np.float32,shape = (batch_size,matrix_length,height,width,in_channel)) 
    y = tf.placeholder(np.float32,shape = (batch_size,class_num)) 
    keep_prob = tf.placeholder(tf.float32)

    Weights_Mani = []

    for i in range(depth):
        Weights_Mani.append({
            'W_DCNN1_root':tf.Variable(tf.random_uniform([k,in_channels[i],middle_channels[i]],minval = 0.01, maxval = 0.99)),
            'W_DCNN2_root':tf.Variable(tf.random_uniform([k,middle_channels[i],out_channels[i]],minval = 0.01, maxval = 0.99)),
            'W_res_wFW_root':tf.Variable(tf.random_uniform([in_channels[i]+out_channels[i],out_channels[i]],minval = 0.01, maxval = 0.99)),
          })

    ############# Chol connect
    W2_1 = tf.Variable(tf.random_normal([n*(n+1)//2 * out_channels[depth-1], class_num],stddev=np.sqrt(2./(class_num*n*(n+1)//2))))
    b2_1 = tf.Variable(tf.random_normal([1, class_num],stddev=np.sqrt(2./class_num)))

    tf.keras.backend.set_learning_phase(True)
    CNNRNNcell = [CNNRNNCell(num_layer = CNN_num_layer, kernel_shape = CNN_kernel_shape , batch_size = batch_size ,
                        matrix_size = matrix_size ,in_channel= in_channel, out_channel=out_channel ,
                        reduced_spatial_dim=reduced_spatial_dim , beta = beta , keep_prob = keep_prob )]
    cells = tf.nn.rnn_cell.MultiRNNCell(CNNRNNcell)
    initial_state = cells.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cells,X,initial_state=initial_state , dtype = np.float32)
    output_series = outputs
    ######################## 

    layer_ = tf.reshape(output_series , [batch_size, matrix_length , n * n , 1])

    for i in range(depth):
        layer_ = res_block(layer_,d0*(2**i),Weights_Mani[i],mode)

    outputs = tf.slice(layer_,[0,matrix_length-1,0,0],[-1,1,-1,-1])
    outputs = tf.reshape(outputs,[batch_size,n*n,out_channels[i]])
    outputs = tf.transpose(outputs,[0,2,1])
    outputs = tf.reshape(outputs,[batch_size*out_channels[i],n,n])

    output_series =  Chol_de ( outputs, n,batch_size*out_channels[i] )
    output_series = tf.reshape(output_series,[batch_size,out_channels[i]*n*(n+1)/2])
    output_series = tf.keras.layers.BatchNormalization()(output_series)
    output_series = tf.nn.dropout(output_series, keep_prob)
    predict_label = tf.add( tf.matmul ( output_series, W2_1 ), b2_1 ) 

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
         logits = predict_label,
         labels = y
    ))

    correct_prediction = tf.equal(tf.argmax(predict_label, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.control_dependencies([add_global]):
        opt = tf.train.AdadeltaOptimizer(learning_rate)
        train_step = opt.minimize(loss)

    batch_num_idx = range(batch_num)
    k_fold = KFold(n_splits=10)
    final_acc_fold = np.zeros((10,1))
    data = []
    label = []


    if not os.path.isfile('./data/UCF11/processed_data/prodata.npy'):
        for idx in range(batch_num):
            print (idx)
            data_batch_in,label_batch_in = readUCFdata(idx,'./data/UCF11/processed_data/',matrix_length,sample_rate)
            data.append(data_batch_in)
            label.append(label_batch_in)

        np.save('./data/UCF11/processed_data/prodata.npy',np.asarray(data))
        np.save('./data/UCF11/processed_data/prolabel.npy',np.asarray(label))
    else:
        data = np.load('./data/UCF11/processed_data/prodata.npy')
        label = np.load('./data/UCF11/processed_data/prolabel.npy')

    test_acc_ave = 0
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.Session() as sess:
        final_acc = 0.
        co = 0
        for tr_indices, ts_indices in k_fold.split(batch_num_idx):
            sess.run(tf.global_variables_initializer())
            print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

            for epoch in range(epoch_num):
                start_time = time.time()
                train_acc = 0.
                train_loss = 0.
                for batch_idx in tr_indices:
                    data_batch_in = data[batch_idx]
                    label_batch_in = label[batch_idx]

                    _, loss_, acc_ = sess.run([train_step,loss,accuracy],
                             feed_dict={
                                   X:data_batch_in,
                                   y:label_batch_in,
                                   keep_prob:0.75
                                    })
                    train_acc = train_acc + acc_
                    train_loss = train_loss + loss_
                train_acc = train_acc / len(tr_indices)
                train_loss = train_loss/len(tr_indices)
                test_acc = 0.
                for batch_idx in ts_indices:
                    data_batch_in = data[batch_idx]
                    label_batch_in = label[batch_idx]
                    loss_, acc_ = sess.run([loss,accuracy],
                                 feed_dict={
                                       X:data_batch_in,
                                       y:label_batch_in,
                                       keep_prob:1.
                                        })
                    test_acc = test_acc + acc_
                test_acc = test_acc / len(ts_indices)
                test_acc_ave = test_acc_ave + test_acc
                if epoch % 20 == 0:
                    print ('Train Accuracy is : ' , train_acc , ' in Epoch : ' , epoch)
                    print ('Train Loss is : ' , train_loss)
                    print ('Time per epoch : ' , time.time()-start_time)
                    print ('Test Accuracy is : ' , test_acc_ave/20.)
                    print (' ')
                    test_acc_ave = 0.
            # pdb.set_trace()
            final_acc_fold[co] = 0.
            for batch_idx in ts_indices:
                data_batch_in = data[batch_idx]
                label_batch_in = label[batch_idx]
                loss_, acc_ = sess.run([loss,accuracy],
                             feed_dict={
                                   X:data_batch_in,
                                   y:label_batch_in,
                                   keep_prob:1.
                                    })
                final_acc_fold[co] = final_acc_fold[co] + 1.0*acc_/len(ts_indices)
                print(loss_,acc_)
            print('After kth fold' , final_acc_fold[co])
            final_acc = final_acc + final_acc_fold[co]*1.0/10
            co += 1
        print(final_acc)
        np.save('UCF11_result.npy',final_acc_fold)