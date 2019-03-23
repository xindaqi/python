import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cifar10_input
import tarfile
from six.moves import urllib
import os
import sys


# 图路径
LOG_DIR = "./logs/standard16"

# 批量数据大小  
batch_size = 512
# 每轮训练数据的组数，每组为一batchsize  
s_times = 20
# 学习率
learning_rate = 0.0001

# Xavier初始化方法 
# 卷积权重(核）初始化 
def init_conv_weights(shape, name): 
	weights = tf.get_variable(name=name, shape=shape, dtype=tf.float32,  initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
	return weights 

# 全连接权重初始化 
def init_fc_weights(shape, name):
	weights = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
	return weights

# 偏置初始化 
def init_biases(shape, name): 
	biases = tf.Variable(tf.random_normal(shape),name=name, dtype=tf.float32) 
	return biases

# 卷积 
# 参数：输入张量,卷积核，偏置，卷积核在高和宽维度上移动的步长
# 卷积核:使用全0填充,padding='SAME' 
def conv2d(input_tensor, weights, biases, s_h, s_w):
	conv = tf.nn.conv2d(input_tensor, weights, [1, s_h, s_w, 1], padding='SAME') 
	return tf.nn.relu(conv + biases)
# 池化 
# 参数：输入张量，池化核高和宽，池化核在高，宽维度上移动步长 
# 池化窗口:使用全0填充,padding='SAME'
def max_pool(input_tensor, k_h, k_w, s_h, s_w):
	return tf.nn.max_pool(input_tensor, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='SAME') 
# 全链接 
# 参数：输入张量，全连接权重，偏置 
def fullc(input_tensor, weights, biases): 
	return tf.nn.relu_layer(input_tensor, weights, biases)
# 使用tensorboard对网络结构进行可视化的效果较好 
# 输入占位节点
# images 输入图像 
with tf.name_scope("source_data"):
	images = tf.placeholder(tf.float32, [batch_size, 224 ,224 ,3])
	# 图像标签
	labels = tf.placeholder(tf.int32, [batch_size]) 
	# 正则 
	keep_prob = tf.placeholder(tf.float32)

# 第一组卷积
# input shape (batch_size, 224, 224, 3)
# output shape (batch_size, 224, 224, 64) 
with tf.name_scope('conv_gp_1'): 
	# conv3-64
	cw_1 = init_conv_weights([3, 3, 3, 64], name='conv_w1') 
	cb_1 = init_biases([64], name='conv_b1') 
	conv_1 = conv2d(images, cw_1, cb_1, 1, 1)
	# conv3-64
	cw_2 = init_conv_weights([3, 3, 64, 64], name='conv_w2')
	cb_2 = init_biases([64], name='conv_b2')
	conv_2 = conv2d(conv_1, cw_2, cb_2, 1, 1)
	
# 最大池化 窗口尺寸2x2,步长2
# input shape (batch_size, 224, 224, 64) 
# output shape (batch_size, 112, 112, 64) 
with tf.name_scope("pool_1"):
	pool_1 = max_pool(conv_2, 2, 2, 2, 2)

# 第二组卷积
# input shape (batch_size, 112, 112, 64)
# output shape (batch_size, 112, 112, 128)
with tf.name_scope('conv_gp2'):
	# conv3-128 
	cw_3 = init_conv_weights([3, 3, 64, 128], name='conv_w3') 
	cb_3 = init_biases([128], name='conv_b3') 
	conv_3 = conv2d(pool_1, cw_3, cb_3, 1, 1)
	# conv3-128 
	cw_4 = init_conv_weights([3, 3, 128, 128], name='conv_w4') 
	cb_4 = init_biases([128], name='conv_b4') 
	conv_4 = conv2d(conv_3, cw_4, cb_4, 1, 1)
	

# 最大池化 窗口尺寸2x2,步长2
# input shape (batch_size, 112, 112, 128)
# output shape (batch_size, 56, 56, 128)
with tf.name_scope("pool_2"):
	pool_2 = max_pool(conv_4, 2, 2, 2, 2)

# 第三组卷积
# input shape (batch_size, 56, 56, 128)
# output shape (batch_size, 56, 56, 256)
with tf.name_scope('conv_gp_3'):
	# conv3-256 
	cw_5 = init_conv_weights([3, 3, 128, 256], name='conv_w5') 
	cb_5 = init_biases([256], name='conv_b5') 
	conv_5 = conv2d(pool_2, cw_5, cb_5, 1, 1) 
	# conv3-256
	cw_6 = init_conv_weights([3, 3, 256, 256], name='conv_w6') 
	cb_6 = init_biases([256], name='conv_b6') 
	conv_6 = conv2d(conv_5, cw_6, cb_6, 1, 1)
	# conv3-256
	cw_7 = init_conv_weights([3, 3, 256, 256], name='conv_w7') 
	cb_7 = init_biases([256], name='conv_b7') 
	conv_7 = conv2d(conv_6, cw_7, cb_7, 1, 1)
	
# 最大池化 窗口尺寸2x2,步长2
# input shape (batch_size, 56, 56, 256)
# output shape (batch_size, 28, 28, 256)
with tf.name_scope("pool_3"):
	pool_3 = max_pool(conv_7, 2, 2, 2, 2) 


# 第四组卷积
# input shape (batch_size, 28, 28, 256)
# output shape (batch_size, 28, 28, 512)
with tf.name_scope('conv_gp_4'): 
	# conv3-512
	cw_8 = init_conv_weights([3, 3, 256, 512], name='conv_w8') 
	cb_8 = init_biases([512], name='conv_b8') 
	conv_8 = conv2d(pool_3, cw_8, cb_8, 1, 1) 
	# conv3-512
	cw_9 = init_conv_weights([3, 3, 512, 512], name='conv_w9') 
	cb_9 = init_biases([512], name='conv_b9') 
	conv_9 = conv2d(conv_8, cw_9, cb_9, 1, 1)
	# conv3-512
	cw_10 = init_conv_weights([3, 3, 512, 512], name='conv_w10') 
	cb_10 = init_biases([512], name='conv_b10') 
	conv_10 = conv2d(conv_9, cw_10, cb_10, 1, 1)


# 最大池化 窗口尺寸2x2,步长2
# input shape (batch_size, 28, 28, 512)
# output shape (batch_size, 14, 14, 512)
with tf.name_scope("pool_4"):
	pool_4 = max_pool(conv_10, 2, 2, 2, 2)


# 第五组卷积　conv3-256 conv3-256 
# input shape (batch_size, 14, 14, 512)
# output shape (batch_size, 14, 14, 512)
with tf.name_scope('conv_gp_5'): 
	# conv3-512
	cw_11 = init_conv_weights([3, 3, 512, 512], name='conv_w11') 
	cb_11 = init_biases([512], name='conv_b11') 
	conv_11 = conv2d(pool_4, cw_11, cb_11, 1, 1) 
	# conv3-512
	cw_12 = init_conv_weights([3, 3, 512, 512], name='conv_w12') 
	cb_12 = init_biases([512], name='conv_b12') 
	conv_12 = conv2d(conv_11, cw_12, cb_12, 1, 1)
	# conv3-512
	cw_13 = init_conv_weights([3, 3, 512, 512], name='conv_w13') 
	cb_13 = init_biases([512], name='conv_b13') 
	conv_13 = conv2d(conv_12, cw_13, cb_13, 1, 1)



# 最大池化 窗口尺寸2x2,步长2
# input shape (batch_size, 14, 14, 512)
# output shape (batch_size, 7, 7, 512)
with tf.name_scope("pool_5"):
	pool_5 = max_pool(conv_13, 2, 2, 2, 2)
	
# 转换数据shape
# input shape (batch_size, 7, 7, 512)
# reshape_conv14 (batch_size, 25088)
reshape_conv13 = tf.reshape(pool_5, [batch_size, -1])
# n_in = 25088
n_in = reshape_conv13.get_shape()[-1].value 


# 第一个全连接层
with tf.name_scope('fullc_1'):
	# (n_in, 4096) = (25088, 4096)
	fw14 = init_fc_weights([n_in, 4096], name='fullc_w14') 
	# (4096, )
	fb14 = init_biases([4096], name='fullc_b14')
	# (batch_size, 25088) x (25088, 4096)
	# (batch_size, 4096)
	activation1 = fullc(reshape_conv13, fw14, fb14) 
# dropout正则 
drop_act1 = tf.nn.dropout(activation1, keep_prob) 
# 第二个全连接层
with tf.name_scope('fullc_2'): 
	# (4096, 4096)
	fw15 = init_fc_weights([4096, 4096], name='fullc_w15') 
	# (4096, )
	fb15 = init_biases([4096], name='fullc_b15') 
	# (batch_size, 4096) x (4096, 4096)
	# (batch_size, 4096) 
	activation2 = fullc(drop_act1, fw15, fb15) 
# dropout正则 
drop_act2 = tf.nn.dropout(activation2, keep_prob) 

# 第三个全连接层
with tf.name_scope('fullc_3'):
	# (4096, 1000)
	fw16 = init_fc_weights([4096, 1000], name='fullc_w16') 
	# (1000, )
	fb16 = init_biases([1000], name='full_b16') 
	# [batch_size, 4096] x [4096, 1000] 
	# [batch_size, 1000]
	logits = tf.add(tf.matmul(drop_act2, fw16), fb16) 
	output = tf.nn.softmax(logits)
# 交叉熵
with tf.name_scope("cross_entropy"):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) 
	cost = tf.reduce_mean(cross_entropy,name='Train_Cost') 
	tf.summary.scalar("cross_entropy", cost)
# 优化交叉熵
with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 准确率
# with tf.name_scope("accuracy"):
# # 用来评估测试数据的准确率 # 数据labels没有使用one-hot编码格式,labels是int32 
#  	def accuracy(labels, output): 
#  		labels = tf.to_int64(labels) 
#  		pred_result = tf.equal(labels, tf.argmax(output, 1)) 
#  		accu = tf.reduce_mean(tf.cast(pred_result, tf.float32))
#  		return accu
 # train_images = 处理后的输入图像
 # train_labels = 处理后的输入图像标签
# 训练		
def training(max_steps, s_times, keeprob, display):
	with tf.Session() as sess:
		# 保存图
		write = tf.summary.FileWriter(LOG_DIR, sess.graph)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		# 协程:单线程实现并发
		coord = tf.train.Coordinator()
		# 使用线程及队列,加速数据处理
		threads = tf.train.start_queue_runners(coord=coord)
		# 训练周期
		for i in range(max_steps): 
			# 小组优化
			for j in range(s_times): 
				start = time.time() 
				batch_images, batch_labels = sess.run([train_images, train_labels]) 
				opt = sess.run(optimizer, feed_dict={images:batch_images, labels:batch_labels, keep_prob:keeprob}) 
				every_batch_time = time.time() - start 
			c = sess.run(cost, feed_dict={images:batch_images, labels:batch_labels, keep_prob:keeprob}) 
			# 保存训练模型路径
			ckpt_dir = './vgg_models/vggmodel.ckpt'
			# 保存训练模型
			saver = tf.train.Saver()
			saver.save(sess,save_path=ckpt_dir,global_step=i)
			# 定步长输出训练结果
			if i % display == 0: 
				samples_per_sec = float(batch_size) / every_batch_time 
				print("Epoch {}: {} samples/sec, {} sec/batch, Cost : {}".format(i+display, samples_per_sec, every_batch_time, c)) 
		# 线程阻塞
		coord.request_stop()
		# 等待子线程执行完毕
		coord.join(threads)

	

if __name__ == "__main__":
	
	training(5000, 5, 0.7, 10)