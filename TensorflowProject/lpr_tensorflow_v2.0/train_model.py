import tensorflow as tf 
import network_model 
from input_data import OCRIter
import numpy as np
import os
import time

batch_size = 8
image_h = 72
image_w = 272
learning_rate = 0.0001
count = 30000
num_label = 7
channels = 3

LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists("./model"):
    os.makedirs("./model")

with tf.name_scope("source_data"):
    '''output data:[batch_size, 36, 136, 3]
    [batch_size, height, width, channels]
    '''
    inputs = tf.placeholder(tf.float32, [batch_size, image_h, image_w, channels], name="p-inputs")
    labels = tf.placeholder(tf.int32, [batch_size, num_label], name="p-labels")
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.image("input", inputs)

def get_batch():
    data_batch = OCRIter(batch_size, image_h, image_w)
    image_batch, label_batch = data_batch.iter()
    image_batch_ = np.array(image_batch)
    label_batch_ = np.array(label_batch)
    return image_batch_, label_batch_ 

train_logits_1, train_logits_2, train_logits_3, train_logits_4, train_logits_5, train_logits_6, train_logits_7 = network_model.inference(inputs, keep_prob)

train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7 = network_model.losses(train_logits_1, train_logits_2, train_logits_3, train_logits_4, train_logits_5, train_logits_6, train_logits_7, labels)

train_op_1, train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7 = network_model.train(train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7, learning_rate)

train_acc = network_model.evaluation(train_logits_1, train_logits_2, train_logits_3, train_logits_4, train_logits_5, train_logits_6, train_logits_7, labels)


summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))


def train_with_reset_graph():
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        train_logits_1, train_logits_2, train_logits_3, train_logits_4, train_logits_5, train_logits_6, train_logits_7 = network_model.inference(inputs, keep_prob)

        train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7 = network_model.losses(train_logits_1, train_logits_2, train_logits_3, train_logits_4, train_logits_5, train_logits_6, train_logits_7, labels)

        train_op_1, train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7 = network_model.train(train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7, learning_rate)

        train_acc = network_model.evaluation(train_logits_1, train_logits_2, train_logits_3, train_logits_4, train_logits_5, train_logits_6, train_logits_7, labels)

        with tf.Session(graph=g) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            # x_batch, y_batch = get_batch()
            # print("data:{}, label: {}".format(x_batch, y_batch))
            for step in range(count):
                x_batch, y_batch = get_batch()
                time_1 = time.time()
                # print("data:{}, label: {}".format(type(x_batch), type(y_batch)))
                feed_dict = {inputs: x_batch, labels: y_batch, keep_prob: 0.5}
                _, _, _, _, _, _, _, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, acc, summary = sess.run([train_op_1,train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7, train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7, train_acc, summary_op], feed_dict)
                time_cost = time.time() - time_1
                loss_all = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
                ckpt_dir = "./model/lpr.ckpt"
                if not os.path.exists("./model"):
                    os.makedirs("./model")
                saver = tf.train.Saver()
                if step % 10 == 0:
                    print("loss1:{:.2f}, loss2:{:.2f}, loss3:{:.2f}, loss4:{:.2f}, loss5:{:.2f}, loss6:{:.2f}, loss7: {:.2f}".format(loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7))
                    print("Total loss: {:.2f}, accuracy: {:.2f}, steps: {}, time cost: {}".format(loss_all, acc, step, time_cost))
                    
                if step % 1000 == 0 or (step+1) == count:
                    saver.save(sess, save_path=ckpt_dir, global_step=step)
                summary_writer.add_summary(summary, step)
        summary_writer.close()

def train_without_reset_graph():
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        x_batch, y_batch = get_batch()
        # print("data:{}, label: {}".format(x_batch, y_batch))
        for step in range(count):
            x_batch, y_batch = get_batch()
            time_1 = time.time()
            # print("data:{}, label: {}".format(type(x_batch), type(y_batch)))
            feed_dict = {inputs: x_batch, labels: y_batch, keep_prob: 0.5}
            _, _, _, _, _, _, _, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, acc, summary = sess.run([train_op_1,train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7, train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7, train_acc, summary_op], feed_dict)
            time_cost = time.time() - time_1
            loss_all = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
            ckpt_dir = "./model/lpr.ckpt"
            summary_writer.add_summary(summary, step)
            print("step: {}, time cost: {}".format(step, time_cost))
            if not os.path.exists("./model"):
                os.makedirs("./model")
            if step % 10 == 0:
                print("loss1:{:.2f}, loss2:{:.2f}, loss3:{:.2f}, loss4:{:.2f}, loss5:{:.2f}, loss6:{:.2f}, loss7: {:.2f}".format(loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7))
                print("Total loss: {:.2f}, accuracy: {:.2f}, steps: {}, time cost: {}".format(loss_all, acc, step, time_cost))
                
            if step % 10 == 0 or (step+1) == count:
                saver = tf.train.Saver()
                saver.save(sess, save_path=ckpt_dir, global_step=step)        
    summary_writer.close()
def train_without_context_session():
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    for step in range(count):
        x_batch, y_batch = get_batch()
        time_1 = time.time()
        # print("data:{}, label: {}".format(type(x_batch), type(y_batch)))
        feed_dict = {inputs: x_batch, labels: y_batch, keep_prob: 0.5}
        _, _, _, _, _, _, _, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, acc, summary = sess.run([train_op_1,train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7, train_loss_1, train_loss_2, train_loss_3, train_loss_4, train_loss_5, train_loss_6, train_loss_7, train_acc, summary_op], feed_dict)
        time_cost = time.time() - time_1
        loss_all = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
        ckpt_dir = "./model/lpr.ckpt"
        
        saver = tf.train.Saver()
        if step % 10 == 0:
            print("loss1:{:.2f}, loss2:{:.2f}, loss3:{:.2f}, loss4:{:.2f}, loss5:{:.2f}, loss6:{:.2f}, loss7: {:.2f}".format(loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7))
            print("Total loss: {:.2f}, accuracy: {:.2f}, steps: {}, time cost: {}".format(loss_all, acc, step, time_cost))
                
        if step % 10 == 0 or (step+1) == count:
            saver.save(sess, save_path=ckpt_dir, global_step=step)
        summary_writer.add_summary(summary, step)
    sess.close()
    
if __name__ == "__main__":
    train_without_reset_graph()
