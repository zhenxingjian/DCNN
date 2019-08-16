import tensorflow as tf
import numpy as np
import pdb

class CNNRNNCell(tf.contrib.rnn.RNNCell):
    """
    Implements a CNN layer like a RNN to share parameters.
    It will compute the SPD matrix from CNN layer.
    """

    def __init__(self , num_layer , kernel_shape , batch_size , matrix_size , in_channel , out_channel , reduced_spatial_dim , beta , keep_prob = 1.0 ):
        '''
        kernel_shape is list of list(size 3, width, height, outchannel, like [5,5,15])
        '''

        self._batch_size = batch_size
        self._matrix_size = matrix_size
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._reduced_spatial_dim = reduced_spatial_dim
        self._beta = beta
        self._num_layer = num_layer
        self._kernel_shape = kernel_shape
        self._keep_prob = keep_prob
        assert num_layer == len(kernel_shape)
        assert kernel_shape[num_layer-1][2] == out_channel
        assert matrix_size == out_channel+1


    @property
    def state_size(self):
        return int(self._matrix_size * self._matrix_size)

    @property
    def output_size(self):
        return int(self._matrix_size * self._matrix_size)

    def __call__(self, inputs, state, scope=None):
        batch_size = self._batch_size
        n = self._matrix_size
        in_channel = self._in_channel
        out_channel = self._out_channel
        reduced_spatial_dim = self._reduced_spatial_dim
        beta = self._beta
        num_layer = self._num_layer
        kernel_shape = self._kernel_shape
        keep_prob = self._keep_prob

        with tf.variable_scope(scope or type(self).__name__):

            Weights_cnn = []
            kernel_out_channel = in_channel
            for layer_idx in range(num_layer):
                kernel_width = kernel_shape[layer_idx][0]
                kernel_height = kernel_shape[layer_idx][1]
                kernel_in_channel = kernel_out_channel
                kernel_out_channel = kernel_shape[layer_idx][2]

                Weights_cnn.append(tf.get_variable('W'+str(layer_idx),[kernel_width,kernel_height,kernel_in_channel,kernel_out_channel],
                                                    initializer = tf.random_normal_initializer(stddev=1e-2), 
                                                    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-2),
                                                    dtype = np.float32))
            
            P1 = inputs

            cov_mat = None

            for layer_idx in range(num_layer):
                C1_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(P1,Weights_cnn[layer_idx],[1,1,1,1],'SAME'))
                C1 = tf.nn.relu(C1_bn)
                C1 = tf.nn.dropout(C1, keep_prob)
                P1 = tf.nn.max_pool(C1,[1,2,2,1],[1,2,2,1],'SAME')


            P2 = tf.transpose(P1,[0,3,2,1])
            Fl = tf.reshape(P2,[batch_size,out_channel,reduced_spatial_dim])
            mean_batch = tf.reduce_mean(Fl,2)   #batch_size x out_channel
            mean_tensor = tf.tile(tf.expand_dims(mean_batch,axis=2),[1,1,reduced_spatial_dim]) #batch_size x out_channel x reduced_spatial_dim
            Fl_m = tf.subtract(Fl,mean_tensor)

            mean_batch = tf.expand_dims(mean_batch,axis=2)
            mean_cov = tf.matmul(mean_batch,mean_batch,transpose_b = True)

            cov_feat = tf.add(tf.matmul(Fl_m, Fl_m, transpose_b=True), beta*beta*mean_cov)

            cov_feat = tf.concat([cov_feat, beta*mean_batch],axis=2)

            mean_batch_t = tf.concat([beta*mean_batch, tf.constant([1.],shape=[batch_size,1,1])],axis=1)
            mean_batch_t = tf.transpose(mean_batch_t,[0,2,1])

            cov_feat = tf.concat([cov_feat, mean_batch_t],axis=1)
            cov_mat = cov_feat

            output = tf.reshape(cov_mat,[batch_size,n*n])
            # pdb.set_trace()
            out_state = state

            return (output, out_state)
