import numpy as np
import tensorflow as tf
import os

import matplotlib.pyplot as plt
tf.reset_default_graph()

MODEL_SAVE_PATH = "./models"
MODEL_NAME = "rnnmodel.ckpt"

'''隐藏层节点'''
HIDDEN_SIZE = 30
'''层数'''
NUM_LAYERS = 2
'''训练序列长度'''
TIMESTEPS = 10
'''训练步数'''
TRAINING_STEPS = 10000
'''bactch'''
BATCH_SIZE = 32
'''训练数据个数'''
TRAINING_EXAMPLES = 10000
'''测试数据个数'''
TESTING_EXAMPLES = 1000
'''采样间隔'''
SAMPLE_GAP = 0.01

def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)
    ])
    
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    
    if not is_training:
        return predictions, None, None
    
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),optimizer="Adagrad",
        learning_rate = 0.1
    )
    return predictions, loss, train_op
  
       
g_params = tf.Graph()
with g_params.as_default():
    X = tf.placeholder(tf.float32, [None, 1, 10])
    y = tf.placeholder(tf.float32, [None, 1])
    predictions, loss, train_op = lstm_model(X, [0.0], False)
    
def load_model():
    outputs = []
    with tf.Session(graph=g_params) as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("./models")
        model_path = ckpt.model_checkpoint_path
        saver.restore(sess, model_path)
        for i in range(TESTING_EXAMPLES):
            input_data = np.expand_dims(test_X[i], 0)
#             print("input data: {}".format(input_data.shape))
            pre = sess.run(predictions, feed_dict={X: input_data})
            outputs.append(pre)
#             print("prediction value: {}".format(pre))
    outputs = np.array(outputs).squeeze()
    plt.figure()
    plt.plot(outputs, label='predictions', marker='x', color='r')
#     plt.plot(y, label='real_sin', marker='|', color='b')
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
#     train_model()
    load_model()
    


