import tensorflow as tf 
import os 
import time
from os.path import join
import matplotlib.pyplot as plt

def max_num(image_path):
    file_lists = os.listdir(image_path)
    numbers = []
    if file_lists:
        for file_list in file_lists:
            number, _ = os.path.splitext(file_list)
            numbers.append(int(number))
        max_number = max([x for x in numbers])
    else:
        max_number = 0
    return max_number

def parse_function(filenames):
    img_bytes = tf.read_file(filenames)
    img_decode = tf.image.decode_jpeg(img_bytes, channels=3)
    return img_decode


def reshape_image(image_path, save_path, max_number, sess):
    '''Reshape image to specify size.

    :params image_path: source image path
    :params save_path: save path for resized image
    :params max_number: max image name number for next batch image name
    :params sess: tensorflow Session

    '''
    imgs_name = os.listdir(image_path)
    png = imgs_name[0].lower().endswith("png")
    imgs_path = [join(image_path, f) for f in imgs_name]
    imgs_num = len(imgs_path)
    '''Image name queue.'''
    imgs_queue = tf.data.Dataset.from_tensor_slices(imgs_path)
    '''Map image queue and process image with parse_function.'''
    imgs_map = imgs_queue.map(parse_function)
    '''Iterator image Tensor value.'''
    img_decode = imgs_map.make_one_shot_iterator().get_next()
    for i in range(imgs_num):
        img_type = img_decode.dtype
        if img_decode.dtype != tf.float32:
            img_decode = tf.image.convert_image_dtype(img_decode, dtype=tf.float32)
        img_decode = tf.image.resize_images(img_decode, [128, 128], method=0)
        if img_decode.dtype == tf.float32:
            img_decode = tf.image.convert_image_dtype(img_decode, dtype=tf.uint8)
        img_value = sess.run(img_decode)
        plt.figure(figsize=(1.28, 1.28))
        plt.imshow(img_value)
        plt.axis("off")
        plt.savefig(save_path+"/{}.jpg".format(max_number+i+1), format="jpg")
        print("Processing {} image.".format(max_number+i+1))
        plt.close("all")

if __name__ == "__main__":
    with tf.Session() as sess:
        if not os.path.exists("./resized_images"):
            os.makedirs("./resized_images")
        save_path = "./resized_images"
        start_time = time.time()
        '''Create Coordinator.'''
        coord = tf.train.Coordinator()
        '''Open therads and fill with Coordinator.'''
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        images_path = './test_images'
        '''directory list: ['images_2', 'images_1']'''
        dirs_list = os.listdir(images_path)
        '''directory path: ['./test_images/images_2', './test_images/images_1']'''
        dirs_path = [join(images_path, f) for f in dirs_list]
        for dir_path in dirs_path:
            max_number = max_num("./resized_images")
            reshape_image(dir_path, save_path, max_number, sess)
        '''Request Coordinator.'''    
        coord.request_stop()
        '''Threads was locked until coordinator exective finished.'''
        coord.join(threads)
        end_time = time.time()
        time_cost = end_time - start_time
        print("Time costed: {}s".format(time_cost))
		