import tensorflow as tf 


# batch_size = 32
# image_h = 36
# image_w = 136
# channels = 3
# label_len = 7

def init_weights_biases(name_w, name_b, shape):
    '''Initial network parameters.
    Args:
        name_w: weights name
        name_b: biases name
        shape: [filter_h, filter_w, current_deep, next_deep]
    Returns:
        weights: weights tensor with spacial shape
        biases: biases tensor with special shape
    '''
    weights = tf.get_variable(name=name_w, shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable(name=name_b, shape=[shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    return weights, biases

def conv2d(input_tensor, ksize, strides, pad, name_w, name_b):
    '''Convolution calculate.
    Args:
        input_tensor: image tensor
        name_w: weights name
        name_b: biases name
        ksize: kernel size [filter_h, filter_w, current_depth, next_depth]
        strides: filter window sliding step, [1, height, width, 1]
        padding: padding type namely padding or not
    Returns:
        conv: convolution result with relu
    '''
    weights = tf.get_variable(name=name_w, shape=ksize, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable(name=name_b, shape=[ksize[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input_tensor, weights, strides=strides, padding=pad)
    conv = tf.nn.relu(conv + biases)
    return conv

def max_pooling(input_tensor, ksize, strides, pad):
    '''Max pooling convolutional result.
    Args:
        input_tensor: convolutional result
        ksize: kernel size [1, height, width, 1]
        strides: filter window sliding step, [1, height, width, 1]
        pad: padding type namely padding or not 
    Returns:
        max_pool: max pooling result
    '''
    max_pool = tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=pad)   
    return max_pool 

def fullc(input_tensor, wsize, name_w, name_b):
    '''Full connection calculate.
    Args:
        input_tensor: convolutional input
        ksize: weights size
        strides: filter window sliding step
    Returns:
        fullc: full connection result
    '''
    weights = tf.get_variable(name=name_w, shape=wsize, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable(name=name_b, shape=[wsize[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    # fullc = tf.nn.relu_layer(input_tensor, weights, biases)
    fullc = tf.matmul(input_tensor, weights) + biases
    return fullc

def small_basic_block(input_tensor, ksize, strides, pad):
    '''Convolution calculate.
    Args:
        input_tensor: image tensor
        name_w: weights name
        name_b: biases name
        ksize: kernel size [filter_h, filter_w, current_depth, next_depth], [1, 1, 64, 64]
        strides: filter window sliding step, [1, height, width, 1], [1, 1, 1, 1]
        padding: padding type namely padding or not
    Returns:
        conv: convolution result with relu
    '''
    conv_s1 = conv2d(input_tensor, [ksize[0],ksize[1],ksize[2],ksize[2]/4], strides, pad, "a", "b")
    conv_s2 = conv2d(conv_s1, [3,1,ksize[2]/4,ksize[2]/4], strides, pad, "c", "d")
    conv_s3 = conv2d(conv_s2, [1,3,ksize[2]/4,ksize[2]/4], strides, pad, "e", "f")
    conv_s4 = conv2d(conv_s3 , [ksize[0],ksize[1],ksize[2]/4,ksize[2]], strides, pad, "g", "h")
    return conv_s4


def inference(inputs, keep_prob):
    with tf.name_scope("conv_1"):
        '''output data:[batch_size, 36, 136, 64]'''
        conv_1 = conv2d(inputs, [3,3,3,32], [1,1,1,1], "VALID", "cw_1", "cb_1")

    with tf.name_scope("conv_2"):
        '''output data:[batch_size, 36, 136, 64]'''
        conv_2 = conv2d(conv_1, [3,3,32,32], [1,1,1,1], "VALID", "cw_2", "cb_2")

    with tf.name_scope("max_pool_1"):
        '''output data:[batch_size, 36, 136, 64]'''
        pooling_1 = max_pooling(conv_2, [1,2,2,1], [1,2,2,1], "VALID")

    with tf.name_scope("conv_3"):
        '''output data:[batch_size, 36, 136, 64]'''
        conv_3 = conv2d(pooling_1, [3,3,32,64], [1,1,1,1], "VALID", "cw_3", "cb_3")

    with tf.name_scope("conv_4"):
        '''output data:[batch_size, 36, 136, 64]'''
        conv_4 = conv2d(conv_3, [3,3,64,64], [1,1,1,1], "VALID", "cw_4", "cb_4")

    with tf.name_scope("max_pool_2"):
        '''output data:[batch_size, 36, 136, 64]'''
        pooling_2 = max_pooling(conv_4, [1,2,2,1], [1,2,2,1], "VALID")

    with tf.name_scope("conv_5"):
        '''output data:[batch_size, 36, 136, 64]'''
        conv_5 = conv2d(pooling_2, [3,3,64,128], [1,1,1,1], "VALID", "cw_5", "cb_5")

    with tf.name_scope("conv_6"):
        '''output data:[batch_size, 36, 136, 64]'''
        conv_6 = conv2d(conv_5, [3,3,128, 128], [1,1,1,1], "VALID", "cw_6", "cb_6")

    with tf.name_scope("max_pool_3"):
        '''output data:[batch_size, 36, 136, 64]'''
        pooling_1 = max_pooling(conv_6, [1,2,2,1], [1,2,2,1], "VALID")

    with tf.name_scope("fullc_1"):
        output_shape = pooling_1.get_shape()
        flatten_1 = output_shape[1].value*output_shape[2].value*output_shape[3].value
        reshape_output = tf.reshape(pooling_1, [-1, flatten_1])
        fc_1 = tf.nn.dropout(reshape_output, keep_prob)

    with tf.name_scope("fullc_21"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_21 = fullc(fc_1, [flatten, 65], "fw2_1", "fb2_1")

    with tf.name_scope("fullc_22"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_22 = fullc(fc_1, [flatten, 65], "fw2_2", "fb2_2")

    with tf.name_scope("fullc_23"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_23 = fullc(fc_1, [flatten, 65], "fw2_3", "fb2_3")

    with tf.name_scope("fullc_24"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_24 = fullc(fc_1, [flatten, 65], "fw2_4", "fb2_4")

    with tf.name_scope("fullc_25"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_25 = fullc(fc_1, [flatten, 65], "fw2_5", "fb2_5")

    with tf.name_scope("fullc_26"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_26 = fullc(fc_1, [flatten, 65], "fw2_6", "fb2_6")

    with tf.name_scope("fullc_27"):
        # flatten = output_shape[1].value*output_shape[2].value*output_shape[3].value
        flatten = reshape_output.get_shape()[-1].value
        fc_27 = fullc(fc_1, [flatten, 65], "fw2_7", "fb2_7")
    return fc_21, fc_22, fc_23, fc_24, fc_25, fc_26, fc_27

def losses(logits_1, logits_2, logits_3, logits_4,logits_5, logits_6, logits_7, labels):
    labels = tf.convert_to_tensor(labels, tf.int32)
    with tf.name_scope("loss_1"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=labels[:,0])
        loss_1 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_1", loss_1)
    
    with tf.name_scope("loss_2"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=labels[:,1])
        loss_2 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_2", loss_2)
    
    with tf.name_scope("loss_3"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=labels[:,2])
        loss_3 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_3", loss_3)

    with tf.name_scope("loss_4"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=labels[:,3])
        loss_4 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_4", loss_4)
    
    with tf.name_scope("loss_5"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=labels[:,4])
        loss_5 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_5", loss_5)

    with tf.name_scope("loss_6"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_6, labels=labels[:,5])
        loss_6 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_6", loss_6)

    with tf.name_scope("loss_7"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_7, labels=labels[:,6])
        loss_7 = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_7", loss_7)
    return loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7


def train(loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, learning_rate):
    with tf.name_scope("optimizer_1"):
        train_op_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
    
    with tf.name_scope("optimizer_2"):
        train_op_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)

    with tf.name_scope("optimizer_3"):
        train_op_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_3)

    with tf.name_scope("optimizer_4"):
        train_op_4 = tf.train.AdamOptimizer(learning_rate).minimize(loss_4)

    with tf.name_scope("optimizer_5"):
        train_op_5 = tf.train.AdamOptimizer(learning_rate).minimize(loss_5)

    with tf.name_scope("optimizer_6"):
        train_op_6 = tf.train.AdamOptimizer(learning_rate).minimize(loss_6)

    with tf.name_scope("optimizer_7"):
        train_op_7 = tf.train.AdamOptimizer(learning_rate).minimize(loss_7)

    return train_op_1, train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7

def evaluation(logits_1, logits_2, logits_3, logits_4,logits_5, logits_6, logits_7, labels):
    '''shape:(8,65)'''
    # print("shape of logits_1: {}".format(logits_1.shape))
    '''shape all logits:(7,8,65)'''
    '''shape: (56, 65)'''
    logits_all = tf.concat([logits_1, logits_2, logits_3, logits_4,logits_5, logits_6, logits_7], 0)
    # print("shape of logits all: {}".format(logits_all.shape))

    '''shape: (8,7)'''
    labels = tf.convert_to_tensor(labels, tf.int32)

    '''shape: (56, 1)'''
    labels_all = tf.reshape(tf.transpose(labels), [-1])
    # print("shape of labels all: {}".format(labels_all.shape))
    with tf.name_scope("accuracy"):
        correct = tf.nn.in_top_k(logits_all, labels_all, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar("accuracy", accuracy)
    return accuracy












    
