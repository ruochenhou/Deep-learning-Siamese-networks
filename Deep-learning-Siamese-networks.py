#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:48:53 2019



SIAMESE NETWORK - ASSESSMENT 2 - IFN680

@authors:   Jui-Chieh Hsiao (Arthur)-n9326774
            Ruochen Hou n9604278
            Andrea del Pilar Rivera Pena n10056858

"""
####################################IMPORTING LIBRARIES##################################################################################################
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,Dropout,MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
######################################### HEADER ########################################################################################################
'''
This assessment was done to implement a Siamese Network. We have used a dataset called "fashion_mnist".  
 as per instructions in the assignment, we have use keras.datasets.fashion_mnist.load_data to load the dataset.
Then we had to split the dataset in the following manner:

train=> (x_train), (y_train) 80% of the labels ["top","trouser","pullover","coat","sandal","ankle boot"] for training
test1=> (x_test1), (y_test1) 20% of the labels ["top","trouser","pullover","coat","sandal","ankle boot"]for testing
test2=> (x_only_test2), (y_only_test2) 100% testing of the labels ["dress","sneaker","bag","shirt"] for testing only.
test3=> (x_validation_test3), (y_validation_test3)= #["top","trouser","pullover","coat","sandal","ankle boot"] union ["dress","sneaker","bag","shirt"].

We have used the following functions to buil the Networks:

1. To create the Siamese network                     def create_siamese_network
2. To create the base network                        def create_base_network(input_shape)
3. To run the Siamese network                        def run_siamese_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets,
                                                                          x_validation, y_validation, optimizer)
4. To run the base network                           def run_base_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets,
                                                                          x_validation, y_validation, optimizer)
5. To create the pairs for each dataset              def create_pairs(x, digit_indices, num_classes)
6. To find the Euclidean output shape                def eucl_dist_output_shape(shapes)
7. To find the Euclidean Distance                    def euclidean_distance(vects)
8. To calculate the contrastive loss function        def contrastive_loss(y_true, y_pred)
9. To testing the loos function                      def testing_loss(y_true, y_pred)
10.To calculate the accuracy                         def accuracy(y_true, y_pred)
11.To compute classification accuracy                def compute_accuracy(y_true, y_pred)
12.To plot the linear graph                          def plot_result(history,epochs,name,y_max)
13.To display the results of siamese network         def dispaly_siamese_network()
14.To display the results of base network            def dispaly_base_network()
15.To star this program                              def main()
'''

def dataset_class():
    # train_test =["top","trouser","pullover","coat","sandal","ankle boot"]
    train_test = [0,1,2,4,5,9]
    
    # only_testing =["dress","sneaker","bag","shirt"]
    only_testing = [3,7,8,6]
    
    # validation total_class =["top","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
    total_class = [0,1,2,3,4,5,6,7,8,9]
    return train_test,only_testing,total_class
    
def data_preparation():
    # loading the dataset fashion_mnist
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Concatenating the values of "x" and the labels of "y"
    x_combine = np.concatenate([x_train, x_test], axis = 0) # combine data for 'x' first
    y_combine = np.concatenate([y_train, y_test], axis = 0) # then combine label for 'y'
    
####################################### PREPARING THE DATASETS(train, test1, test2, test3) ####################################################################
    
    # the labels for the fashion_mnist are:
    
       # 0 T-shirt/top
       # 1 Trouser
       # 2 Pullover
       # 3 Dress
       # 4 Coat
       # 5 Sandal
       # 6 Shirt
       # 7 Sneaker
       # 8 Bag
       # 9 Ankle boot
       
    # Label classification#
    
    train_test,only_testing,total_class = dataset_class()
    
    
    #Creating empty arrays  to populate with values obtained from the labels train_test 
    #dataset = ["top","trouser","pullover","coat","sandal","ankle boot"]
    x_train_full = []
    y_train_full = []
    
    # creating empty arrays to populate with values obtained from the labels only_test2 dataset= ["dress","sneaker","bag","shirt"]
    x_only_test2= []
    y_only_test2= []
    
    for index in range(len(y_combine)):
        if y_combine[index] in train_test:
            x_train_full.append(x_combine[index])
            y_train_full.append(y_combine[index])
        else:
            x_only_test2.append(x_combine[index])
            y_only_test2.append(y_combine[index])
    
    ###########(x_train), (y_train) 80% of the labels ["top","trouser","pullover","coat","sandal","ankle boot"]##############################################
    
    x_train = np.array(x_train_full) 
    y_train = np.array(y_train_full)
    
    ###########(x_test1), (y_test1) 20% of the labels ["top","trouser","pullover","coat","sandal","ankle boot"]##############################################
    
    x_train, x_test1 , y_train, y_test1 = train_test_split(x_train, y_train, test_size=0.2, random_state=5) 
    
    
    ###########(x_only_test2),(y_only_test2) 100% testing of the labels ["dress","sneaker","bag","shirt"]####################################################
    
    x_only_test2 = np.array(x_only_test2)
    y_only_test2 = np.array(y_only_test2)
    
    ###########(x_validation_test3), (y_validation_test3) union of arrays. test3= (test1 U only_test2) ######################################################
    
    x_test3 = np.concatenate([x_test1, x_only_test2], axis = 0) 
    y_test3 = np.concatenate([y_test1, y_only_test2], axis = 0)
    return x_train,y_train,x_test1,y_test1,x_only_test2,y_only_test2,x_test3,y_test3


#Function to create pairs and populate the pairs and label arrays
def create_pairs(x, digit_indices, num_classes):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [0, 1]
    return np.array(pairs), np.array(labels)

##########################Euclidean Distance ############################################################################################################

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

#####################Contrative Loss Function############################################################################################################
def contrastive_loss(y_true, y_pred):

    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)
#####################Testing Loss Function############################################################################################################
# For testing the loss function by adjusting the margin value 
    
def testing_loss(y_true, y_pred):

    margin = 5
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)

################## ACCURACY FUNCTION#####################################################################################################################

def accuracy(y_true, y_pred):
   
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

def compute_accuracy(y_true, y_pred):
    
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)

################################### CREATING THE BASE_NETWORK  ########################################################################################
#FUNCTIONAL API

def create_base_network(input_shape):

    input_layer = Input(input_shape)
    hidden1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(input_layer)  
    hidden2 = Conv2D(64, (3, 3), activation='relu')(hidden1)
    hidden3 = MaxPooling2D(pool_size=(2, 2))(hidden2)
    hidden4 = Flatten()(hidden3)
    hidden5 = Dense(128, activation='relu')(hidden4)
    output = Dense(128,activation='relu')(hidden5)
    return Model(input_layer, output)

################################### CREATING THE SIAMESE_NETWORK  ########################################################################################
#FUNCTIONAL API

def create_siamese_network(input_shape):

    input_layer = Input(input_shape)
    hidden1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(input_layer)  
    hidden2 = Conv2D(64, (3, 3), activation='relu')(hidden1)
    hidden3 = MaxPooling2D(pool_size=(2, 2))(hidden2)
    hidden4 = Flatten()(hidden3)
    hidden5 = Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),  
                    bias_regularizer=regularizers.l1(0.01))(hidden4)
    hidden6 = Dropout(0.1)(hidden5)
    output = Dense(128,activation='relu')(hidden6)
    return Model(input_layer, output)


############## RUNNING THE BASE_NETWORK ####################################################################################################

def run_base_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets,
                            x_validation, y_validation, optimizer):
    
    #pairing data
    tr_digit_indices = [np.where(y_train == i)[0] for i in train_sets]
    tr_pairs, tr_y = create_pairs(x_train, tr_digit_indices,len(train_sets))
    te_digit_indices = [np.where(y_test == i)[0] for i in test_sets]
    te_pair, te_y = create_pairs(x_test, te_digit_indices, len(test_sets))
    va_digit_indices = [np.where(y_validation == i)[0] for i in train_sets]
    va_pairs, va_y = create_pairs(x_validation, va_digit_indices, len(train_sets))

    #rehsape
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    input_shape = (img_rows,img_cols,1)
    tr_pairs = tr_pairs.reshape(tr_pairs.shape[0],2,img_rows,img_cols,1)
    te_pair = te_pair.reshape(te_pair.shape[0],2,img_rows,img_cols,1)
    va_pairs = va_pairs.reshape(va_pairs.shape[0],2,img_rows,img_cols,1)
    
    # convert to float32 and rescale between 0 and 1
    tr_pairs = tr_pairs.astype('float32')
    te_pair = te_pair.astype('float32')
    va_pairs = va_pairs.astype('float32')
    tr_pairs /= 255
    te_pair /= 255
    va_pairs /=255

    base_network = create_base_network(input_shape)
    
    #Define the tensors for the two input images using the same base_network 
    image_a = Input(shape=input_shape)
    image_b = Input(shape=input_shape)

    #We will distribute the weights equally between the two branches.
    siamese_Branch_a = base_network(image_a)
    siamese_Branch_b = base_network(image_b)
    
    #Processing the distances
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([siamese_Branch_a, siamese_Branch_b])

    model = Model([image_a, image_b], distance)
    
    #training data  
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,epochs=epochs,verbose=1,
              validation_data=([va_pairs[:, 0], va_pairs[:, 1]], va_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)

    y_pred = model.predict([te_pair[:, 0], te_pair[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    
    #evaluation 
    score = model.evaluate([te_pair[:, 0], te_pair[:, 1]], te_y, verbose=0)
    return tr_acc, te_acc, history,score 

############## RUNNING THE SIAMESE_NETWORK ####################################################################################################

def run_siamese_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets,
                            x_validation, y_validation, optimizer):
    
    #pairing data
    tr_digit_indices = [np.where(y_train == i)[0] for i in train_sets]
    tr_pairs, tr_y = create_pairs(x_train, tr_digit_indices,len(train_sets))
    te_digit_indices = [np.where(y_test == i)[0] for i in test_sets]
    te_pair, te_y = create_pairs(x_test, te_digit_indices, len(test_sets))
    va_digit_indices = [np.where(y_validation == i)[0] for i in train_sets]
    va_pairs, va_y = create_pairs(x_validation, va_digit_indices, len(train_sets))

    #rehsape
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    input_shape = (img_rows,img_cols,1)
    tr_pairs = tr_pairs.reshape(tr_pairs.shape[0],2,img_rows,img_cols,1)
    te_pair = te_pair.reshape(te_pair.shape[0],2,img_rows,img_cols,1)
    va_pairs = va_pairs.reshape(va_pairs.shape[0],2,img_rows,img_cols,1)
    
    #normalising
    tr_pairs = tr_pairs.astype('float32')
    te_pair = te_pair.astype('float32')
    va_pairs = va_pairs.astype('float32')
    tr_pairs /= 255
    te_pair /= 255
    va_pairs /=255

    cnn_network = create_siamese_network(input_shape)
    
    #Define the tensors for the two input images using the same base_network 
    image_a = Input(shape=input_shape)
    image_b = Input(shape=input_shape)

    #We will distribute the weights equally between the two branches.
    siamese_Branch_a = cnn_network(image_a)
    siamese_Branch_b = cnn_network(image_b)
    
    #Processing the distances
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([siamese_Branch_a, siamese_Branch_b])

    model = Model([image_a, image_b], distance)
    
    #training data  
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,epochs=epochs,verbose=1,
              validation_data=([va_pairs[:, 0], va_pairs[:, 1]], va_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)

    y_pred = model.predict([te_pair[:, 0], te_pair[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    
    #evaluation 
    score = model.evaluate([te_pair[:, 0], te_pair[:, 1]], te_y, verbose=0)
    return tr_acc, te_acc, history,score 

 
############## PLOTTING THE TRAINING AND VALIDATION VS TIME#############################################################################################
 # Plot training & validation accuracy values

def plot_result(history,epochs,name):
    # Plot training & validation accuracy values
    print('\nThe Loss and Accuracy in {}:\n'.format(name))
    plt.plot(history.history['accuracy'],'-o')
    plt.plot(history.history['val_accuracy'],'-o')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    x_ticks = np.arange(0, epochs+1,1)
    plt.xticks(x_ticks)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'],'-o')
    plt.plot(history.history['val_loss'],'-o')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    x_ticks = np.arange(0, epochs+1,1)
    plt.xticks(x_ticks)
    plt.legend(['train', 'val_loss'], loc='upper left')
    plt.show()
    
########################### Dispaly THE SIAMESE_NETWORK t#############################################################################################
def dispaly_siamese_network():
    
    train_test,only_testing,total_class = dataset_class()
    epochs = 20
    optimizer = RMSprop()
    x_train,y_train,x_test1,y_test1,x_only_test2,y_only_test2,x_test3,y_test3 = data_preparation()
    x_validation = x_test1
    y_validation = y_test1
    train_sets = train_test
    
    
    print('-----------------------------Siamese model------------------------------ ')
    print('-----------------------------Test1------------------------------ ')
    test_sets = train_test
    tr_acc, te_acc,information,score = run_siamese_network(x_train, y_train, x_test1, y_test1, epochs,
                                                         train_sets, test_sets, x_validation,y_validation,optimizer)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Evaluation test loss: %.2f' % score[0])
 
    
    print('-----------------------------only_Test2------------------------------ ')
    test_sets = only_testing
    tr_acc, te_acc,information,score = run_siamese_network(x_train, y_train, x_only_test2, y_only_test2, epochs,
                                                         train_sets, test_sets, x_validation,y_validation,optimizer)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Evaluation test loss: %.2f' % score[0])

    
    print('-----------------------------Test3------------------------------ ')
    test_sets = total_class
    tr_acc, te_acc,information,score = run_siamese_network(x_train, y_train, x_test3, y_test3, epochs,
                                                         train_sets, test_sets, x_validation,y_validation,optimizer)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Evaluation test loss: %.2f' % score[0])
    plot_result(information, epochs, "Siamese network")   
############################## Dispaly the base netework result#############################################################################################
def dispaly_base_network():
   
    train_test,only_testing,total_class = dataset_class()
    epochs = 20
    optimizer = RMSprop()
    x_train,y_train,x_test1,y_test1,x_only_test2,y_only_test2,x_test3,y_test3 = data_preparation()
    x_validation = x_test1
    y_validation = y_test1
    train_sets = train_test

    print('-----------------------------Base model------------------------------ ')
    print('-----------------------------Test1------------------------------ ')
    test_sets = train_test
    tr_acc, te_acc,information,score = run_base_network(x_train, y_train, x_test1, y_test1, epochs,
                                                         train_sets, test_sets, x_validation,y_validation,optimizer)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Evaluation test loss: %.2f' % score[0])
    
    print('-----------------------------only_Test2------------------------------ ')
    test_sets = only_testing
    tr_acc, te_acc,information,score = run_base_network(x_train, y_train, x_only_test2, y_only_test2, epochs,
                                                         train_sets, test_sets, x_validation,y_validation,optimizer)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Evaluation test loss: %.2f' % score[0])
    
    print('-----------------------------Test3------------------------------ ')
    test_sets = total_class
    tr_acc, te_acc,information,score = run_base_network(x_train, y_train, x_test3, y_test3, epochs,
                                                         train_sets, test_sets, x_validation,y_validation,optimizer)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Evaluation test loss: %.2f' % score[0])
    plot_result(information, epochs, "Base network")   
    
################################# Main function#############################################################################################
def main():
    
   dispaly_base_network()
   dispaly_siamese_network()
 

if __name__=='__main__':
       main()

