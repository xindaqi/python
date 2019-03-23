import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf 

v1 = tf.constant([125.0], shape=[1], name='v1')
v2 = tf.constant([125.0], shape=[1], name='v2')
startTime = time.time()
result = v1 + v2
endTime = time.time()
timeConsuming = endTime - startTime
config = tf.ConfigProto(log_device_placement=True)

with tf.Session(config=config) as sess:
	print("CPU calculator result is : {}".format(sess.run(result)))
	print("CPU calculator time consuming is: {}".format(timeConsuming))

