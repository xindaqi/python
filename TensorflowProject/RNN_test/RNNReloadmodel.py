import numpy as np
import tensorflow as tf
import os

import matplotlib.pyplot as plt
tf.reset_default_graph()

MODEL_SAVE_PATH = "./models"
MODEL_NAME = "rnnmodel.ckpt"

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32
TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y, is_training):
    '''Create lstm model structure.
    
    Args:
        X: source data for train
        y: sourde data label for optimize
        is_training: bool, flag for training or not 

    Returns:
        predictions: predicted value
        loss: loss value between predicted value and labels
        train_op: optimize operation 
    '''
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
    '''Train lstm model structure.
    Args:
        sess: tensorflow session
        train_X: source data for train
        train_y: source data labels
    
    Returns:
        None
    '''
    X = tf.placeholder(tf.float32, [BATCH_SIZE, 1, 10])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 1])
    predictions, loss, train_op = lstm_model(X, y, True)
             
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    x_input, y_input = ds.make_one_shot_iterator().get_next()
    x_input, y_input = sess.run([x_input, y_input])
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss], feed_dict={X: x_input, y: y_input})
        if i % 1000 == 0:
            print("train step:{}, loss: {}".format(str(i), str(l)))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))  

def train_model():
    '''Train model engine.'''
    with tf.Session() as sess:
        train(sess, train_X, train_y)

'''Graph structure.'''            
g_params = tf.Graph()
with g_params.as_default():
    X = tf.placeholder(tf.float32, [None, 1, 10])
    y = tf.placeholder(tf.float32, [None, 1])
    predictions, loss, train_op = lstm_model(X, [0.0], False)
    
def load_model():
    '''Load model and predicting value.'''
    pre_outputs = []
    with tf.Session(graph=g_params) as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("./models")
        model_path = ckpt.model_checkpoint_path
        saver.restore(sess, model_path)
        for i in range(TESTING_EXAMPLES):
            input_data = np.expand_dims(test_X[i], 0)
            pre = sess.run(predictions, feed_dict={X: input_data})
            pre_outputs.append(pre)
    outputs = np.array(pre_outputs).squeeze()
    plt.figure()
    plt.plot(outputs, label='predictions', marker='x', color='r')
    plt.plot(test_y, label='real_sin', marker='|', color='b')
    plt.legend()
    plt.show()
    
test_start = (TRAINING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
test_end = test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP

train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
    
if __name__ == "__main__":
#     train_model()
    load_model()
    