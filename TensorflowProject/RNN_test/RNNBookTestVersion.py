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

'''auto number'''
def auto_number(dir_path):
    image_nums = os.listdir(dir_path)
    numbers = []
    if image_nums:
        for image_num in image_nums:
            number, _ = os.path.splitext(image_num)
            numbers.append(int(number))
        max_number = max([x for x in numbers])
    else:
        max_number = 0

    return max_number

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

def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()
    
    with tf.variable_scope("model", reuse=True):
        predictions, loss, train_op = lstm_model(X, y, True)
        
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print("train step:{}, loss: {}".format(str(i), str(l)))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
            
def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
        predictions = []
        labels = []
        for i in range(TESTING_EXAMPLES):
            p, l = sess.run([prediction, y])
            predictions.append(p)
            labels.append(l)
    
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels)**2).mean(axis=0))
    print("Mean Square Error is : {}".format(rmse))
    
#     plt.figure()
#     plt.plot(predictions, label='predictions')
#     plt.plot(labels, label="real value")
#     plt.legend()
#     plt.show()
    numbers = auto_number("./train_result")
    
    plt.figure()
    plt.plot(predictions, label='predictions', marker='x', color='r')
    plt.plot(labels, label='real_sin', marker='|', color='b')
    plt.legend()
    plt.grid()
    plt.savefig("./train_result/{}.png".format(numbers+1),format="png")
    plt.show()




    
test_start = (TRAINING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
test_end = test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP

train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES+TIMESTEPS, dtype=np.float32)))

# with tf.variable_scope("model"):
#     _, loss, train_op = lstm_model(X, y, True)
ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
X, y = ds.make_one_shot_iterator().get_next()
    
with tf.variable_scope("model"):
    predictions, loss, train_op = lstm_model(X, y, True)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("Before RNN Training")
    run_eval(sess, test_X, test_y)
    train(sess, train_X, train_y)
    print("After RNN Training")
    run_eval(sess, test_X, test_y)
    