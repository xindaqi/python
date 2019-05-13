import tensorflow as tf 
import network_model 
from input_data import OCRIter
import numpy as np
import os

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

with tf.name_scope("source_data"):
    '''output data:[batch_size, 36, 136, 3]
    [batch_size, height, width, channels]
    '''
    inputs = tf.placeholder(tf.float32, [batch_size, image_h, image_w, channels], name="inputs")
    labels = tf.placeholder(tf.int32, [batch_size, num_label], name="labels")
    keep_prob = tf.placeholder(tf.float32)

def get_batch():
    data_batch = OCRIter(batch_size, image_h, image_w)
    image_batch, label_batch = data_batch.iter()
    image_batch = np.array(image_batch)
    label_batch = np.array(label_batch)
    return image_batch, label_batch 

train_logits_1, train_logits_2, train_logits_3, train_logits_4, \
    train_logits_5, train_logits_6, train_logits_7 = network_model.inference(inputs, keep_prob)

train_loss_1, train_loss_2, train_loss_3, train_loss_4, \
    train_loss_5, train_loss_6, train_loss_7 = network_model.losses(train_logits_1, train_logits_2, train_logits_3, train_logits_4, \
        train_logits_5, train_logits_6, train_logits_7, labels)

train_op_1, train_op_2, train_op_3, train_op_4, \
    train_op_5, train_op_6, train_op_7 = network_model.train(train_loss_1, train_loss_2, train_loss_3, train_loss_4, \
        train_loss_5, train_loss_6, train_loss_7, learning_rate)

train_acc = network_model.evaluation(train_logits_1, train_logits_2, train_logits_3, train_logits_4, \
    train_logits_5, train_logits_6, train_logits_7, labels)


summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
# summary_op = tf.summary.merge_all(tf.get_collection(tf.GraphKeys.SUMMARIES))

if __name__ == "__main__":
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        # x_batch, y_batch = get_batch()
        # print("data:{}, label: {}".format(x_batch, y_batch))
        for step in range(count):
            x_batch, y_batch = get_batch()
            # print("data:{}, label: {}".format(type(x_batch), type(y_batch)))
            feed_dict = {inputs: x_batch, labels: y_batch, keep_prob: 0.5}
            _, _, _, _, _, _, _, loss_1, loss_2, loss_3, loss_4, \
                loss_5, loss_6, loss_7, acc, summary = sess.run([train_op_1,\
                    train_op_2, train_op_3, train_op_4, train_op_5, train_op_6, train_op_7,\
                        train_loss_1, train_loss_2, train_loss_3, train_loss_4, \
                            train_loss_5, train_loss_6, train_loss_7, train_acc, summary_op], \
                                feed_dict=feed_dict)
            
            loss_all = loss_1 + loss_2 + loss_3 + loss_4 + \
                loss_5 + loss_6 + loss_7
            ckpt_dir = "./model/lpr.ckpt"
            if not os.path.exists("./model"):
                os.makedirs("./model")
            saver = tf.train.Saver()
            if step % 10 == 0:
                print("loss1:{:.2f}, loss2:{:.2f}, loss3:{:.2f}, loss4:{:.2f}, loss5:{:.2f}, loss6:{:.2f}, loss7: {:.2f}".format(loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7))
                print("Total loss: {:.2f}, accuracy: {:.2f}".format(loss_all, acc))
                
            if step % 1000 == 0 | (step+1) == count:
                saver.save(sess, save_path=ckpt_dir, global_step=step)
            summary_writer.add_summary(summary, step)
        coord.request_stop()
        coord.join(threads)
    summary_writer.close()


