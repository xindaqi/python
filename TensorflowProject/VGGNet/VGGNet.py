import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cifar10_input
import tarfile
from six.moves import urllib
import os
import sys



FLAGS = tf.app.flags.FLAGS
LOG_DIR = "./logs/testlogs"
# 模型参数
tf.app.flags.DEFINE_string('data_dir', 'cifa10_data',
							"""Path to the CIFAR-10 data directory.""")



def data_stream():
	# cifar10.maybe_download_and_extract()
	data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	data_directory = FLAGS.data_dir
	print("Data directory: {}".format(data_directory))
	if not os.path.exists(data_directory):
		os.mkdir(data_directory)

	filename = data_url.split('/')[-1]
	filepath = os.path.join(data_directory, filename)
	print("File path: {}".format(filepath))
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>>Downloading {} {}'.format(filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
		statinfo = os.stat(filepath)
		print("Successfully downloaded", filename, statinfo.st_size, 'bytes.')
	extracted_dir_path = os.path.join(data_directory, 'cifar-10-batches-bin')
	if not os.path.exists(extracted_dir_path):
		tarfile.open(filepath, 'r:gz').extractall(data_directory)




#一共训练的两万多次，分了两次，中途保存过一次参数变量
max_steps = 20700
# 小批量数据大小  
batch_size = 512
# 每轮训练数据的组数，每组为一batchsize  
s_times = 20
learning_rate = 0.0001
# self.data_dir = 'cifar10data/cifar-10-batches-bin' 
# 数据所在路径
data_dir = "./cifa10_data/cifar-10-batches-bin"

# Xavier初始化方法 
# 卷积权重(核）初始化 
def init_conv_weights(shape, name): 
	weights = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
	return weights 

# 全连接权重初始化 
def init_fc_weights(shape, name):
	weights = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
	return weights

# 偏置 
def init_biases(shape, name): 
	biases = tf.Variable(tf.random_normal(shape),name=name, dtype=tf.float32) 
	return biases


# 卷积 
# 参数：输入张量,卷积核，偏置，卷积核在高和宽维度上移动的步长 
def conv2d(input_tensor, weights, biases, s_h, s_w):
	conv = tf.nn.conv2d(input_tensor, weights, [1, s_h, s_w, 1], padding='VALID') 
	return tf.nn.relu(conv + biases)


# 池化 
# 参数：输入张量，池化核高和宽，池化核在高，宽维度上移动步长 
def max_pool(input_tensor, k_h, k_w, s_h, s_w):
	return tf.nn.max_pool(input_tensor, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID') 
	
# 全链接 
# 参数：输入张量，全连接权重，偏置 
def fullc(input_tensor, weights, biases): 
	return tf.nn.relu_layer(input_tensor, weights, biases)

with tf.name_scope("source_data"):
# 输入占位节点
# images 输入图像 
	images = tf.placeholder(tf.float32, [batch_size, 24 ,24 ,3])
	# 图像标签
	labels = tf.placeholder(tf.int32, [batch_size]) 
	# 正则 
	keep_prob = tf.placeholder(tf.float32)
	tf.summary.image("input", images, 8)

# 使用作用域对Op进行封装，在使用tensorboard对网络结构进行可视化的效果较好 
# 第一组卷积 conv3-16
# input shape (batch_size, 24, 24, 3)
# output (batch_size, 22, 22, 16) 
with tf.name_scope('conv_gp_1'): 
	cw_1 = init_conv_weights([3, 3, 3, 16], name='conv_w1') 
	cb_1 = init_biases([16], name='conv_b1') 
	conv_1 = conv2d(images, cw_1, cb_1, 1, 1)
	reshape_cgp_1 = tf.reshape(conv_1[0], [16, 22, 22, 1])
	tf.summary.image('conv_gp_1', reshape_cgp_1, 8)

# 最大池化 2x2
# input 30x30x16
# output 15x15x16
# pool_1 = max_pool(conv_1, 2, 2, 2, 2)


# 第二组卷积　conv3-32 
# input (batch_size, 22, 22, 16)
# output (batch_size, 20, 20, 32)
with tf.name_scope('conv_gp2'): 
	cw_2 = init_conv_weights([3, 3, 16, 32], name='conv_w2') 
	cb_2 = init_biases([32], name='conv_b2') 
	conv_2 = conv2d(conv_1, cw_2, cb_2, 1, 1)
	reshape_cgp_2 = tf.reshape(conv_2[0], [32, 20, 20, 1])
	tf.summary.image('conv_gp_2', reshape_cgp_2, 8)

# 最大池化
# input 28x28x32
# output 14x14x32
# pool_2 = max_pool(conv_2, 2, 2, 2, 2)


# 第三组卷积　conv3-64  conv3-64
# input (batch_size, 20, 20, 32)
# output (batch_size, 16, 16, 64)
with tf.name_scope('conv_gp_3'): 
	cw_3 = init_conv_weights([3, 3, 32, 64], name='conv_w3') 
	cb_3 = init_biases([64], name='conv_b3') 
	conv_3 = conv2d(conv_2, cw_3, cb_3, 1, 1) 
	cw_4 = init_conv_weights([3, 3, 64, 64], name='conv_w4') 
	cb_4 = init_biases([64], name='conv_b4') 
	conv_4 = conv2d(conv_3, cw_4, cb_4, 1, 1)
	reshape_cgp_3 = tf.reshape(conv_4[0], [64, 16, 16, 1])
	tf.summary.image('conv_gp_3', reshape_cgp_3, 8)


# 最大池化
# input 5x5x64
# output 3x3x64
# pool_3 = max_pool(conv4, 2, 2, 2, 2) 


# 第四组卷积　conv3-128 conv3-128 
# input (batch_size, 16, 16, 64)
# output (batch_size, 12, 12, 128)
with tf.name_scope('conv_gp_4'): 
	cw_5 = init_conv_weights([3, 3, 64, 128], name='conv_w5') 
	cb_5 = init_biases([128], name='conv_b5') 
	conv_5 = conv2d(conv_4, cw_5, cb_5, 1, 1) 
	cw_6 = init_conv_weights([3, 3, 128, 128], name='conv_w6') 
	cb_6 = init_biases([128], name='conv_b6') 
	conv_6 = conv2d(conv_5, cw_6, cb_6, 1, 1)
	reshape_cgp_4 = tf.reshape(conv_6[0], [128, 12, 12, 1])
	tf.summary.image('conv_gp_4', reshape_cgp_4, 8)


# 此时张量的高和宽为　３ｘ３，继续池化为　２ｘ２
# input (batch_size, 12, 12, 128)
# output (batch_size, 6, 6, 128)
pool_4 = max_pool(conv_6, 2, 2, 2, 2)
reshape_pool_4 = tf.reshape(pool_4[0], [128, 6, 6, 1])
tf.summary.image('pool_4', reshape_pool_4, 8)


# 第五组卷积　conv3-256 conv3-256 
# input (batch_size, 6, 6, 128)
# output (batch_size, 2, 2, 128)
with tf.name_scope('conv_gp_5'): 
	cw_7 = init_conv_weights([3, 3, 128, 128], name='conv_w7') 
	cb_7 = init_biases([128], name='conv_b7') 
	conv_7 = conv2d(pool_4, cw_7, cb_7, 1, 1) 
	cw_8 = init_conv_weights([3, 3, 128, 128], name='conv_w8') 
	cb_8 = init_biases([128], name='conv_b8') 
	conv_8 = conv2d(conv_7, cw_8, cb_8, 1, 1)
	reshape_cgp_5 = tf.reshape(conv_8[0], [128, 2, 2, 1])
	tf.summary.image('conv_gp_5', reshape_cgp_5, 8)


# 转换数据shape
# input batch_size x 2 x 2 x 128
# reshape_conv8 (batch_size, 512)
reshape_conv8 = tf.reshape(conv_8, [batch_size, -1])
# n_in = 512
n_in = reshape_conv8.get_shape()[-1].value 



# 第一个全连接层命名空间 
with tf.name_scope('fullc_1'):
	# 512x256 
	fw9 = init_fc_weights([n_in, 256], name='fullc_w9') 
	# 256x1
	fb9 = init_biases([256], name='fullc_b9')
	# 512x256 
	activation1 = fullc(reshape_conv8, fw9, fb9) 
# dropout正则 
drop_act1 = tf.nn.dropout(activation1, keep_prob) 

with tf.name_scope('fullc_2'): 
	fw10 = init_fc_weights([256, 256], name='fullc_w10') 
	fb10 = init_biases([256], name='fullc_b10') 
	# 512 x 256
	activation2 = fullc(drop_act1, fw10, fb10) 
# dropout正则 
drop_act2 = tf.nn.dropout(activation2, keep_prob) 


with tf.name_scope('fullc_3'): 
	fw11 = init_fc_weights([256, 10], name='fullc_w11') 
	fb11 = init_biases([10], name='full_b11') 
	# 512 x 10
	logits = tf.add(tf.matmul(drop_act2, fw11), fb11) 
	output = tf.nn.softmax(logits)



with tf.name_scope("cross_entropy"):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) 
	cost = tf.reduce_mean(cross_entropy,name='Train_Cost') 
	tf.summary.scalar("cross_entropy", cost)

with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# with tf.name_scope("accuracy"):

# # 用来评估测试数据的准确率 # 数据labels没有使用one-hot编码格式,labels是int32 
# 	def accuracy(labels, output): 
# 		labels = tf.to_int64(labels) 
# 		pred_result = tf.equal(labels, tf.argmax(output, 1)) 
# 		accu = tf.reduce_mean(tf.cast(pred_result, tf.float32))
# 		tf.summary.scalar('accuracy', accu) 
# 		# return accu



merged = tf.summary.merge_all()




# 加载训练batch_size大小的数据，经过增强处理，剪裁，反转，等等
train_images, train_labels = cifar10_input.distorted_inputs(batch_size= batch_size, data_dir= data_dir)

# 加载测试数据，batch_size大小，不进行增强处理
test_images, test_labels = cifar10_input.inputs(batch_size= batch_size, data_dir= data_dir,eval_data= True)

# Training

def train(sess, max_steps, s_times, keeprob, display):
	for i in range(max_steps):
		for j in range(s_times): 
			start = time.time() 
			batch_images, batch_labels = sess.run([train_images, train_labels]) 
			opt = sess.run(optimizer, feed_dict={images:batch_images, labels:batch_labels, keep_prob:keeprob}) 
			every_batch_time = time.time() - start 
		c = sess.run(cost, feed_dict={images:batch_images, labels:batch_labels, keep_prob:keeprob}) 
		# Cost.append(c)
			
		ckpt_dir = './vgg_models/vggmodel.ckpt'
		saver = tf.train.Saver()
		saver.save(sess,save_path=ckpt_dir,global_step=i)

		if i % display == 0: 
			samples_per_sec = float(batch_size) / every_batch_time 
			# format_str = 'Epoch %d: %d samples/sec, %.4f sec/batch, Cost : %.5f'
			print("Epoch {}: {} samples/sec, {} sec/batch, Cost : {}".format(i+display, samples_per_sec, every_batch_time, c)) 
			# print format_str%(i+display, samples_per_sec, every_batch_time, c) 
			# return Cost



def training(max_steps, s_times, keeprob, display):
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		# Cost = []
		
		summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
		# writer.close() 
		for i in range(max_steps): 
			for j in range(s_times): 
				start = time.time() 
				batch_images, batch_labels = sess.run([train_images, train_labels]) 
				opt = sess.run(optimizer, feed_dict={images:batch_images, labels:batch_labels, keep_prob:keeprob}) 
				every_batch_time = time.time() - start 
			summary, c = sess.run([merged, cost], feed_dict={images:batch_images, labels:batch_labels, keep_prob:keeprob}) 
			# Cost.append(c)
			
			ckpt_dir = './vgg_models/vggmodel.ckpt'
			saver = tf.train.Saver()
			saver.save(sess,save_path=ckpt_dir,global_step=i)

			if i % display == 0: 
				samples_per_sec = float(batch_size) / every_batch_time 
				# format_str = 'Epoch %d: %d samples/sec, %.4f sec/batch, Cost : %.5f'
				print("Epoch {}: {} samples/sec, {} sec/batch, Cost : {}".format(i+display, samples_per_sec, every_batch_time, c)) 
				# print format_str%(i+display, samples_per_sec, every_batch_time, c) 
				# return Cost
			summary_writer.add_summary(summary, i)
		coord.request_stop()
		coord.join(threads)

	summary_writer.close()

if __name__ == "__main__":



	training(5000, 5, 0.7, 10)

	# source_data = GetData()
	# source_data.data_stream()

	# with tf.Session() as sess:
	# 	init_op = tf.global_variables_initializer()
	# 	sess.run(init_op)
	# 	writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
	# 	tf.train.start_queue_runners(sess=sess)
		
		# 图片增强处理,时使用了16个线程加速,启动16个独立线程
		# tf.train.start_queue_runners(sess=sess)
		# train_data, train_labels = sess.run([train_images, train_labels])
		# # (512, 24, 24, 3)
		# print("Shape of train data: {}".format(train_data.shape))
		# print("Train data: {}".format(train_data))
		# # (512,)
		# print("Shape of train label: {}".format(train_labels.shape))
		# print("Train label: {}".format(train_labels))


		
		
		# # 加载训练3200次的权重变量
		# # saver = tf.train.Saver()
		# # saver.restore(sess,'./vgg_weights-3200')

		# train(sess, 3200, 5, 0.7, 10)

		# fig,ax = plt.subplots(figsize=(13,6))
		# ax.plot(train_cost)
		# plt.title('Train Cost')
		# plt.grid()
		# plt.show()


# 	# 	# 训练评估 
# 	# 	# train__images, train__labels = sess.run([train_images, train_labels]) 
# 	# 	# train_output = sess.run(output,feed_dict={images:train__images,keep_prob:1.0}) 
# 	# 	# train_accuracy = sess.run(accuracy(train__labels, output=train_output)) 
# 	# 	# 测试评估 
# 	# 	# test__images, test__labels = sess.run([test_images, test_labels]) 
# 	# 	# test_output = sess.run(output, feed_dict={images:test__images, keep_prob:1.0}) 
# 	# 	# test_accuracy = sess.run(accuracy(test__labels, test_output)) 
# 	# 	# print 'train accuracy is: %.7f'%train_accuracy 
# 	# 	# print 'test  accuracy is: %.7f'%test_accuracy


# 	# 	# ckpt_dir = './vgg_weights'
# 	# 	# saver = tf.train.Saver()
# 	# 	# saver.save(sess,save_path=ckpt_dir,global_step=3200)

















