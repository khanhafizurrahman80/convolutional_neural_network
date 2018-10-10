import convolutional_preprocessing as cp
import sys
import pandas as pd
import tensorflow as tf
import convolutional_tensorflow_traditional as ctt
import matplotlib.pyplot as plt

#folder = '/Users/khanhafizurrahman/Desktop/Job_preparation/machine_learning/Classification/Datasets/style'

def main(folder):
    input_image_data_n_arr = cp.input_images(folder)
    input_df = pd.read_csv(folder + '/style.csv')
    print('shape of input df: {shape}'.format(shape= input_df.shape))
    cp.print_total_no_of_unique_products(input_df)
    product_name_target = cp.classLabels(input_df)

    '''fig = plt.figure(figsize=(18,6))

    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[], title=input_df['brand_name'] [i*218] + ' || ' + input_df['product_name'][i*218])
        cp.display_images(input_df['file'][i*218], ax, folder)'''

    input_norm_image_data = cp.normalize_input_array(input_image_data_n_arr) # numpy array
    split_train_test = cp.splitDataset(input_norm_image_data, product_name_target)
    train_feature_data = split_train_test['X_train']
    train_target_data = split_train_test['Y_train']
    test_feature_data = split_train_test['X_test']
    test_target_data = split_train_test['Y_test']

    print ('train feature data shape: {0} \
            \ntrain target data shape: {1} \
            \ntest feature data shape: {2} \
            \ntest test data shape: {3}'.format(train_feature_data.shape, train_target_data.shape, test_feature_data.shape, test_target_data.shape))
    
    Y_train, Y_test = cp.convert_to_one_hot(train_target_data, test_target_data)
    print (Y_train.shape , '\n', Y_test.shape)
    with tf.Session():
        print(Y_train[0:5].eval())
        print(Y_test[0:5].eval())
    
    ctt.model(train_feature_data, Y_train, test_feature_data, Y_test)

if __name__ == '__main__':
    folder_path = str(sys.argv[1])
    main(folder_path)