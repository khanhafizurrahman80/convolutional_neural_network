import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholder for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    print ("inside of the create_placeholders -- file name: convolutional_tensorflow_traditional")
    print ('value of n_H0 {0} \
            \nvalue of n_W0 {1}\
            \nvalue of n_C0  {2}\
            \nvalue of n_y {3}'.format(n_H0, n_W0, n_C0, n_y))
    X = tf.placeholder(tf.float32, [None,n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

def initialize_parameters(filter_size_H0, filter_size_W0, no_input_channel, no_of_filters, name_of_var):
    """
    Concept: initial only weights, bias are handeled by the tensorflow. This weights are only for the conv2d layer.
    shape of the weight: [height_filter_size, width_filter_size, no_input_channel, total_no_of_filters]

    Arguments: 
    filter_size_H0 :: Height of the filter 
    filter_size_W0 :: Width of the filter
    no_input_channel :: 3 for RGB image
    no_of_filter :: number of filters to be operated
    name_of_var -- name of the variable to be assigned in tensorflow

    Returns:
    W -- weight parameter of the convolutional network having defined shape 
    """
    print ("inside of the initialize_parameters -- file name: convolutional_tensorflow_traditional")
    print ('value of filter_size_H0 {0} \
            \nvalue of filter_size_W0 {1}\
            \nvalue of no_input_channel {2}\
            \nvalue of no_of_filters {3}\
            \nvalue of name_of_var {4}'.format(filter_size_H0, filter_size_W0, no_input_channel, no_of_filters, name_of_var))
    tf.set_random_seed(1)
    W = tf.get_variable(name_of_var,[filter_size_H0, filter_size_W0, no_input_channel, no_of_filters], initializer= tf.contrib.layers.xavier_initializer(seed= 0) )
    return W

def forward_propagation(X, parameters, unique_lable_class_no):
    """
    Steps:
        -- Conv2D: stride 1, padding is "SAME"
        -- ReLU
        -- Max pool: Use an n_H0 by n_W0 filter size and an n by n stride, padding is "SAME"
        -- Conv2D: stride 1, padding is "SAME" 
        -- ReLU
        -- Max pool: Use an n_H1 by n_W1 filter size and an n by n stride, padding is "SAME"
        -- Flatten: the previous output from the pooling layer.
        -- FULLYCONNECTED (FC) layer: Apply a fully connected layer without an non-linear activation function  

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionalry containing your parameters "W1", "W2"; the shapes are given in initialize_parameters

    Returns: 
    Z3 -- output of the last LINEAR unit
    """
    print ("inside of the forward_propagation -- file name: convolutional_tensorflow_traditional")
    print ('shape of X {0} \
            \nparameters {1}'.format(X.shape, parameters))
    W1 = parameters['W1']
    W2 = parameters['W2']
    with tf.variable_scope('conv1') as scope:
        Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME")
        print('81: shape of Z1 {0}'.format(Z1.shape))
    with tf.variable_scope('relu1') as scope:
        A1 = tf.nn.relu(Z1)
        print('84: shape of A1 {0}'.format(A1.shape))
    with tf.variable_scope('pool1') as scope:
        P1 = tf.nn.max_pool(A1,ksize= [1,2,2,1], strides = [1,2,2,1],padding="SAME")
        print('87: shape of P1 {0}'.format(P1.shape))
    with tf.variable_scope('conv2') as scope:
        Z2 = tf.nn.conv2d(P1,W2, strides=[1,1,1,1], padding= "SAME")
        print('90: shape of Z2 {0}'.format(Z2.shape))
    with tf.variable_scope('relu2') as scope:
        A2 = tf.nn.relu(Z2)
        print('93: shape of A2 {0}'.format(A2.shape))
    with tf.variable_scope('pool2') as scope:
        P2 = tf.nn.max_pool(A2,ksize=[1,2,2,1], strides= [1,2,2,1], padding='SAME')
        print('96: shape of P2 {0}'.format(P2.shape))
    with tf.variable_scope('flatten') as scope:
        P = tf.contrib.layers.flatten(P2)
        print('99: shape of P {0}'.format(P.shape))
    with tf.variable_scope('fully_connected') as scope:
        Z3 = tf.contrib.layers.fully_connected(P, unique_lable_class_no, activation_fn= None)
        print('102: shape of Z3 {0}'.format(Z3.shape))
    return Z3

def compute_cost(Z3, Y):
    """
    Compute the cost.

    Arguments:
    Z3 -- output of forward propagation (output of the last linear unit), of shape(no_of_unique_class, number_of_examples)
    Y -- one_hot label vector, same shape as Z3

    Returns:
    cost -- Tensor of the cost function
    """
    print ("inside of the compute_cost -- file name: convolutional_tensorflow_traditional")
    print ('shape of Z3 {0} \
            \nshape of Y {1}'.format(Z3.shape, Y.shape))
    with tf.variable_scope('Softmax'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost

def random_mini_batches(X, Y, minibatch_size =64, seed =0):
    '''
    Creates a list of random minibatches from (X, Y)

    Arguments: 
    X -- input data; of sdhape (input size, number of examples)
    Y -- true "label" one hot code vector; shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    '''
    print ("inside of the random_mini_batches -- file name: convolutional_tensorflow_traditional")
    print ('shape of X {0} \
            \nshape of Y {1} \
            \n type of X {2} \
            \n type of Y {3}'.format(X.shape, Y.shape, type(X), type(Y)))
    np.random.seed(seed)
    no_of_training_examples = X.shape[0]
    print ('no of training examples {0}'.format(no_of_training_examples)) 
    mini_batches = []

    # Step 1: shuffle (X,Y)
    permutation = list(np.random.permutation(no_of_training_examples)) # randomly shuffled the array index position np.random.permutation(np.range(3)) returns any combination between 0 to 2 but not using one value twice
    shuffled_X = X[permutation, :, :, :]
    # shuffled_Y = Y[:, permutation].reshape((1,no_of_training_examples))
    shuffled_Y = Y[permutation, :]
    
    # Step 2: partition (shuffled_X, shuffled_Y). minus the end case
    num_complete_minibatches = math.floor(no_of_training_examples/minibatch_size) # no. of mini batches of size mini_batch_size in your partitioning

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * minibatch_size:(k + 1) * minibatch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * minibatch_size:(k + 1) * minibatch_size,:]
        print ('149')
        print (mini_batch_X.shape, mini_batch_Y.shape)
        print (mini_batch_X.dtype, mini_batch_Y.dtype)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if no_of_training_examples % minibatch_size != 0:
        #end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[num_complete_minibatches * minibatch_size:, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * minibatch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
     
def convert_tensor_to_numpy_array(Y_train, Y_test):
    print ("inside of the convert_tensor_to_numpy_array function -- file name: convolutional_tensorflow_traditional")
    Y_train_arr = np.zeros((1747,10))
    Y_test_arr = np.zeros((437, 10))
    with tf.Session().as_default():
        Y_train_arr = Y_train.eval()
        Y_test_arr = Y_test.eval() 
    return Y_train_arr, Y_test_arr


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=10, minibatch_size=64, print_cost= True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, height of image, width of image, no_of_channel)
    Y_train -- test set, of shape (None, no_of_target_class)
    X_test -- training set, of shape (None, height of image, width of image, no_of_channel)
    Y_test -- test set, of shape (None, no_of_target_class)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    print ("inside of the model function -- file name: convolutional_tensorflow_traditional")
    print ('train feature data shape {0} \
            \ntrain target data shape {1} \
            \ntest feature data shape {2} \
            \ntest target data shape {3} '.format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    Y_train_arr, Y_test_arr = convert_tensor_to_numpy_array(Y_train, Y_test)
    print ('type of Y_train_arr {0} \
        \ntype of Y_test_arr {1}'.format(type(Y_train_arr), type(Y_test_arr)))
    tf.set_random_seed(1) # tensorflow seed
    seed = 3 # numpy seed
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train_arr.shape[1]
    costs = []

    # Create placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize paramaeters
    W1 = initialize_parameters(5,5,3,32, "W1")
    W2 = initialize_parameters(5,5,32,64, "W2")
    parameters = {"W1": W1, "W2": W2}

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, int(n_y)) 

    # cost function
    cost = compute_cost(Z3, Y)

    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

    # initialize all the variables globally
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter('Training_conv_FileWriter/')

    tf.summary.scalar('loss', cost)
    merged_summary = tf.summary.merge_all()
    print ('240: type of merged summary {0}'.format(type(merged_summary)))
    # start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # run the initialization
        sess.run(init)
        writer.add_graph(sess.graph)

        #Do the training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            minibatch_cost = 0

            num_minibatches = int(m/minibatch_size) # no of minibatches of size minibatch_size in the train_set
            seed = seed + 1
            minibatches =  random_mini_batches(X_train, Y_train_arr, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                print ('type of minibatch _x{0} \
                        \ntype of minibatch_y{1} \
                        \ntype of merged_summary{2}'.format(type(minibatch_X.dtype), type(minibatch_Y.dtype), type(merged_summary)))
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost/num_minibatches
                summ = sess.run(merged_summary, feed_dict= {X:minibatch_X, Y:minibatch_Y})
                writer.add_summary(summ, num_minibatches)

            end_time= time.time()
            print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")

            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train_arr})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test_arr})
        print ("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        return train_accuracy, test_accuracy, parameters