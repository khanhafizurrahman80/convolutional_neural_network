import pandas as pd
import numpy as np
import os
import image
import tensorflow as tf
from subprocess import check_output
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def input_images(folder):
    onlyfiles = [f for f in os.listdir(folder) if f.endswith(".png")]
    onlyfiles = [x for x in sorted(onlyfiles)]
    final_numpy_array = np.zeros((len(onlyfiles), 150, 150,3))
    print("Working with {0} images".format(len(onlyfiles)))
    not_listed=[]
    for i in range(len(onlyfiles)):
        file_selected = folder + '/' + onlyfiles[i]
        image_data = mpimg.imread(file_selected)
        if image_data.shape[0] == 150:
            image_data = np.delete(image_data, 3, 2) # axis 2 mane hole 3rd dimension ta i.e (150,150,3) er 3
            final_numpy_array[i, :, :, :] = image_data
        else:
            not_listed.append(i)
    print (final_numpy_array.shape)
    return final_numpy_array

def print_total_no_of_unique_products(input_df):
    print('total unique product name {0}, \nUnique Product names {1}'.format(np.unique(input_df.product_name).shape[0], np.unique(input_df.product_name)))
    print('shape of input df: {shape}'.format(shape= input_df.shape ))
    print ('**' * 50)
    print('different type of products {products}'.format(products= np.unique(input_df.product_name)))
    print ('**' * 50)
    print('total number of boots {boots}'.format(boots = input_df[input_df.product_name == 'boots'].shape[0]))
    print ('**' * 50)
    print('total number of bracelet {bracelet}'.format(bracelet = input_df[input_df.product_name == 'bracelet'].shape[0]))
    print ('**' * 50)
    print('total number of earrings {earrings}'.format(earrings = input_df[input_df.product_name == 'earrings'].shape[0]))
    print ('**' * 50)
    print('total number of handbag {handbag}'.format(handbag = input_df[input_df.product_name == 'handbag'].shape[0]))
    print ('**' * 50)
    print('total number of lipstick {lipstick}'.format(lipstick = input_df[input_df.product_name == 'lipstick'].shape[0]))
    print ('**' * 50)
    print('total number of nail_polish {nail_polish}'.format(nail_polish = input_df[input_df.product_name == 'nail polish'].shape[0]))
    print ('**' * 50)
    print('total number of necklace {necklace}'.format(necklace = input_df[input_df.product_name == 'necklace'].shape[0]))
    print ('**' * 50)
    print('total number of ring {ring}'.format(ring = input_df[input_df.product_name == 'ring'].shape[0]))
    print ('**' * 50)
    print('total number of shoes {shoes}'.format(shoes = input_df[input_df.product_name == 'shoes'].shape[0]))
    print ('**' * 50)
    print('total number of watches {watches}'.format(watches = input_df[input_df.product_name == 'watches'].shape[0]))

def classLabels(input_df):
    product_name_target = input_df.product_label.values
    print(product_name_target.shape)
    return product_name_target

def display_images(img_path, ax, folder):
    plt.axis("off")
    plt.imshow(mpimg.imread(folder + '/' +img_path))
    #plt.show() # to show the image uncomment the following line

def normalize_input_array(input_image_data_n_arr):
    '''
    convert RGB images into normalized image
    Argument: input_image_data_n_arr: image containing shape (total_no_images,  height, width, 3)
    Return: images contain value between 0 to 1
    '''
    return input_image_data_n_arr.astype('float32')/255

def convertion_into_gray_images(input_image_data_n_arr):
    '''convert RGB image data into gray image
    Argument: input_image_data_n_arr: image containing shape (total_no_images,  height, width, 3)
    Return: image containing shape (total_no_images,  height, width)
    '''
    input_normalize_image_data_n_arr = normalize_input_array(input_image_data_n_arr)
    return np.dot(input_normalize_image_data_n_arr[..., :3], [.299, 0.587, 0.114])

def splitDataset(inputData, inputLabelData):
    '''
    split the dataset into train data and test data
    inputData: input normalize image data 
    output : dictionary containing separate of train and test data having different portion of feature and label data 
    '''
    X_train, X_test, Y_train, Y_val = train_test_split(inputData, inputLabelData, test_size=0.2)
    return {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_val}

def convert_to_one_hot(Y_train_orig, Y_test_orig):
    '''convert the target image into one hot vector
    Argument:
        Y_train_orig: label data of training image
        Y_test_orig: label data of test image
    Return: a tuple of
        Y_train: one hot vector of train label data
        Y_test_orig: one hot vector of test label data
    '''
    unique_val_of_train_label = np.unique(Y_train_orig)
    unique_val_of_test_label = np.unique(Y_test_orig)
    Y_train = tf.one_hot(Y_train_orig, len(unique_val_of_train_label))
    Y_test = tf.one_hot(Y_test_orig, len(unique_val_of_test_label))
    return (Y_train, Y_test)    

