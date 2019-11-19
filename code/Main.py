import DataImport
import ModelLearning
import timeit
from tensorflow.python.client import device_lib
train_dir_name = '../asl_alphabet_train/asl_alphabet_train'
train_csv_name = '../asl_alphabet_train/asl_alphabet_train.csv'

test_dir_name = '../asl_alphabet_test/asl_alphabet_test'
test_csv_name = '../asl_alphabet_test/asl_alphabet_test.csv'

#Uncomment to create csv
#DataImport.img_to_csv(test_dir_name, test_csv_name)

x_train, y_train = DataImport.get_data(train_csv_name)
print(x_train, y_train)

start = timeit.timeit()

ModelLearning.train_model(x_train, y_train)

end = timeit.timeit()
print(end - start)

print (x_test, y_test)

