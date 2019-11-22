import numpy as np
import os
import csv
from PIL import Image

size = 48, 48
class_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q',
               'R', 'S', 'space' 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Accepts int, returns string value of corresponding class name
def get_class_name(class_num):
    return class_names[int(class_num)]


def get_class_index(c):
    return np.argmax(c)


def img_to_csv(dir_name, csv_name):
    class_names = os.listdir(dir_name)
    print(class_names)

    for c in class_names:
        dir = dir_name + '/' + c

        img_class = np.zeros(len(class_names))
        img_class[class_names.index(c)] = 1

        print("now importing images from class: ", c)

        for file in os.listdir(dir):
            img = Image.open(dir + '/' + file).convert('L')
            img.thumbnail(size, Image.ANTIALIAS)
            img = np.array(img).flatten()/255
            img = np.append(img, img_class)
            with open(csv_name, "a", newline='') as fp:
                wr = csv.writer(fp)
                wr.writerow(img)


def get_data(csv_name):
    num_classes = 29
    num_inputs = 48 * 48

    data_set = np.loadtxt(csv_name, delimiter=',')

    print(data_set.shape)
    x_data = data_set[:, 0:num_inputs]  # extract the pixel values
    y_data = data_set[:, num_inputs:num_inputs + num_classes]# extract the class

    x_data = np.reshape(x_data, (x_data.shape[0], 48, 48, 1))

    y_clean = np.zeros(y_data.shape[0])
    for i, c in enumerate(y_data):
        y_clean[i] = get_class_index(c)

    return x_data, y_clean
