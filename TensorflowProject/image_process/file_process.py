import os 
from os.path import join, abspath, dirname
import shutil
base_dir = abspath(dirname(__name__))
print("base directory: {}".format(base_dir))
# images_path = './train_images'

def copy_file(image_path):
    '''Save image or other file to another folder.
    :params image_path: image path
    '''
    '''Get sub diectory.
    ['images_2', 'images_1']
    '''
    directory_list = os.listdir(images_path)
    print("directory list: {}".format(directory_list))
    '''Glob root directory and sub directory.
    ['./test_images/images_2', './test_images/images_1']
    '''
    dir_path = [join(images_path, f) for f in directory_list]
    print("directory path: {}".format(dir_path))
    '''Extract sub directory file.'''
    file_list = os.listdir(dir_path[0])
    print("file name list: {}".format(file_list))
    '''Glob sub directory and file.
    ['./test_images/images_2/b3.jpg', './test_images/images_2/b4.jpg', 
    './test_images/images_2/b1.jpg', './test_images/images_2/b2.jpg']
    '''
    file_path = [join(dir_path[0], file_name) for file_name in file_list]
    print("file path: {}".format(file_path))
    for file in file_path:
        shutil.copy(file, dir_path[1])

'''Image path where has multi-level directory.'''
images_path = './test_images'
copy_file(images_path)