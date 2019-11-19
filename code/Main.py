import DataImport
import ModelLearning
import timeit
from tensorflow.python.client import device_lib

print ('start')

train_dir_name = '../asl_alphabet_train/asl_alphabet_train'
train_csv_name = '../asl_alphabet_train/asl_alphabet_train.csv'

test_dir_name = '../asl_alphabet_test/asl_alphabet_test'
test_csv_name = '../asl_alphabet_test/asl_alphabet_test.csv'

#Uncomment to create csv
#DataImport.img_to_csv(train_dir_name, train_csv_name)

x_train, y_train = DataImport.get_data(train_csv_name)
print(x_train.shape, y_train.shape)


ModelLearning.train_model(x_train, y_train)


