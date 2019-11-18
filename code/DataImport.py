import numpy as np
import os
import csv
from PIL import Image

train_dir_name = '../asl_alphabet_train/asl_alphabet_train'
train_csv_name = "../asl_alphabet_train/asl_alphabet_train.csv"
class_names = os.listdir(train_dir_name)
size = 48,48


# Accepts int, returns string value of corresponding class name
def get_class_name(class_num):
    return class_names[class_num]


def img_to_csv():

    for c in class_names:
        dir = train_dir_name + '/' + c

        img_class = np.zeros(len(class_names))
        img_class[class_names.index(c)] = 1

        print("now importing images from class: ", c)

        for file in os.listdir(dir):
            img = Image.open(dir + '/' + file).convert('L')
            img.thumbnail(size, Image.ANTIALIAS)
            img = np.array(img).flatten()
            img = np.append(img, img_class)

            with open(train_csv_name, "a", newline='') as fp:
                wr = csv.writer(fp)
                wr.writerow(img)
