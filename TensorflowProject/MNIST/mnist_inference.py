#-*-coding:utf-8-*-
import tensorflow as tf

#定义神经网络结构相关参数
INPUT_NODE = 784#28x28
OUTPUT_NODE = 10#0~9
LAYER1_NODE = 500

#通过tf.get_variable函数获取变量.
#测试时通过保存的模型加载变量取值.
#在变量加载时将滑动平均变量重命名.
#可以直接通过相同的名字在训练时使用变量自身.
#测试时使用变量的滑动平均值.
#该函数将正则化损失嵌入损失集合.
def get_weight_variable(shape, regularizer):
	weights = tf.get_variable("weights", shape,
		initializer=tf.truncated_normal_initializer(stddev=0.1))
	#当给出正则化生成函数时,将当前变量的正则化损失加入名为losses的集合.
	#使用add_to_collection将一个张量加入一个集合.
	#该集合是自定义集合,不在Tensorflow自动管理的集合列表中.
	if regularizer !=None:
		tf.add_to_collection('losses', regularizer(weights))
	return weights

#定义神经网络的前向传播过程.
def inference(input_tensor, regularizer):
	#声明第一层神经网络的变量并完成前向传播过程.
	with tf.variable_scope('layer1'):
		#该处tf.get_variable或tf.Variable无本质区别.
		#在训练或测试中,没有同一个程序多次调用该函数.
		#若在同一个程序中多次调用,在第一次调用之后,需将reuse参数设置为True.
		weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable("biases", [LAYER1_NODE],
			initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
		#声明第二层神经网络变量及前向传播过程.
	with tf.variable_scope('layer2'):
		weights = get_weight_variable(
			[LAYER1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable(
			"biases", [OUTPUT_NODE],
			initializer=tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases
	#返回前向传播结果.
	return layer2


