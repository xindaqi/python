import os
import shutil

def auto_number(dir_path):
    '''Get max number in folder dir_path with its nanme
    and auto number for next saved file.
    
    :params dir_path: files path where contained file named with number

    return:
    :params max_number: maximum number extract from file name
    '''
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

def format_size(bytes_data):
    '''Get file size.
    
    :params bytes_data: input data format is bytes
    return:
    :params G: file size which unit GB
    :params M: file size which unit MB
    :params kb: file size which unit kb
    '''
    try:
        bytes_data = float(bytes_data)
        kb = bytes_data / 1024
    except:
        print("Error format")
        return "error"
    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M /1024
            return "%fG" %(G)
        else:
            return "%fM" %(M)
    else:
        return "%fkb" %(kb)

def get_file_size(file_path):
    '''Calculate file size.
    :params file_path: file path

    return:
    :params file_size: output file size with unit G or M or kb
    '''
    file_size = os.path.getsize(file_path)
    file_size = format_size(file_size)
    return file_size
def get_folder_size(folder_path):
    '''Calculate folder size.
    
    :params folder_path: folder path

    return:
    :params sumsize: folder size with uint G or M or kb
    '''
    sumsize = 0
    try:
        file_name = os.walk(folder_path)
        for root, dirs, files in file_name:
            for file in files:
                size = os.path.get_file_size(folder_path+file)
                sumsize += size
        sumsize = format_size(sumsize)
        return sumsize
    except Exception as err:
        print(err)

