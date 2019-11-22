import DataImport
import ModelLearning
import pickle
import timeit
from tensorflow.python.client import device_lib
from tensorflow import keras

print ('start')

train_dir_name = '../asl_alphabet_train/asl_alphabet_train'
train_csv_name = '../asl_alphabet_train/asl_alphabet_train.csv'

test_dir_name = '../asl_alphabet_test/asl_alphabet_test'
test_csv_name = '../asl_alphabet_test/asl_alphabet_test.csv'

#Uncomment to create csv
#DataImport.img_to_csv(train_dir_name, train_csv_name)
#DataImport.img_to_csv(test_dir_name, test_csv_name)


# --------------------TRAIN------------------------ #

# pull from csv (takes long)
#x_train, y_train = DataImport.get_data(train_csv_name)
x_train, y_train= DataImport.get_data(test_csv_name)

# save array to pickle after pulling from csv
#with open('train.pickle', 'wb') as f:
    #pickle.dump([x_train, y_train], f)

# pull from pickle (is fast)
with open('train.pickle', 'rb') as f:
    x_test, y_test = pickle.load(f)




ModelLearning.train_model(x_train, y_train)

# ---------------------TEST-------------------------- #

#print(x_train.shape, y_train.shape)

model = keras.models.load_model('CNN_POOLING_TESTING.h5')

model.evaluate(x_test, y_test)


