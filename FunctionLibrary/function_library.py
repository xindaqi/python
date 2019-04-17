import os

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