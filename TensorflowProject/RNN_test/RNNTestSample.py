#-*-coding:utf-8-*-
import numpy as np 

X = [1, 2]
state = [0.0, 0.0]

#全连接层内部结构
weights_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
weights_input = np.asarray([0.5, 0.6])
biases = np.asarray([0.1, -0.1])

#全连接层连接输出的参数
weights_output = np.asarray([[1.0], [2.0]])
biases_out = 0.1

#顺序执行循环神经网络前向传播
for i in range(len(X)):
	fc_pretreatment = np.dot(state, weights_state) + X[i]*weights_input + biases
	state = np.tanh(fc_pretreatment)

	#最终输出
	final_output = np.dot(state, weights_output) + biases_out

	#输出每个时刻信息
	print("全连接层预处理(未激活) :",fc_pretreatment)
	print("激活函数处理结果:", state)
	print("最终处理结果:", final_output)