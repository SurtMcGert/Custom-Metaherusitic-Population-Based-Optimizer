# Computational Intelligence Course Work
# imports
import pickle
import os
import numpy as np 
from PIL import Image as im 
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers
from keras.regularizers import l2

# global variables
DATASET_PATH = 'dataset' # the directory that the dataset files are
IMAGE_DIM = 32 # the dimension of all the images is 32 x 32
IMAGE_CHANNELS = 3 # each image is RGB so has 3 channels
NO_OF_CLASSES = 10 # there are 10 classes of images in the dataset
CNN_MODEL_FILE = "cnnModel.h5" # the name of the file storing the weights of the CNN model
MODEL_WITH_ALGORITHM_FILE = "cnnWithAlgorithm.h5" # the name of the file storing the weights of the model after using our optimization algorithm

# function to load a data file into a dictionary containing:
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
# inputs:
# file - path to the file to read
# 
# returns the file as a dictionary
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# function to read the dataset
# inputs:
# folder - the folder that the dataset files are in
# 
# returns two dictionaries (trainingData, testData)
def readData(folder):
    print("reading data")
    files = os.listdir(folder)
    index = 0
    trainingData = {}
    testData = {}
    # loop over all the files in the folder
    while index < len(files):
        filename = files[index]
        path = folder + "/" + filename
        # if the file name starts with "data" then it is a training data batch
        if filename.startswith('data'):
            print("read ", filename)
            dic = unpickle(path)
            for key, value in dic.items():
                try:
                    trainingData[key.decode("UTF-8")].append(value)
                except:
                    trainingData[key.decode("UTF-8")] = [value]

        # otherwise if it starts with "test" then it is the test data
        elif filename.startswith('test'):
            print("read ", filename)
            dic = unpickle(path)
            for key, value in dic.items():
                try:
                    testData[key.decode("UTF-8")].append(value)
                except:
                    testData[key.decode("UTF-8")] = [value]
        index += 1
    return trainingData, testData



# function to save an image for viewing
# inputs:
# imageArr - the 3D array of pixel data containing data in row major order
def saveImage(imageArr):
    print("saving image")
    # show the shape of the array 
    print(imageArr.shape)
    #create an image object using the array
    data = im.fromarray(imageArr)
    # save the output as a PNG file
    data.save('image.png')

# function to build a CNN
# inputs:
# trainingData - the data that will be used to train the model
# 
# returns: a CNN model
def model(trainingData):
    print("making a CNN")
    #define a model
    model = keras.Sequential()

    #add layers to the model
    initializer = keras.initializers.LecunUniform(seed=0)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.00), kernel_initializer=initializer))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.00), kernel_initializer=initializer))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(NO_OF_CLASSES, activation="softmax", kernel_initializer=initializer))


    #build the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    shape = np.shape(trainingData)
    batchSize = shape[0]
    model.build(input_shape=(batchSize, shape[1], shape[2], shape[3]))

    #get a summery of the model
    model.summary()

    return model

# function to train a given model and save its weights to a file
# inputs:
# model - the model to train
# filename - the name of the file to save the weights in
# trainingData - the data used to train the model
# trainingLabels - the labels associated with the training data
# testData - the data used to test the model
# testLabels - the labels associated with the test data
def trainModel(model, filename, trainingData, trainingLabels, testData, testLabels):
    model.fit(trainingData[:, :, :, :], trainingLabels[:], validation_data=(testData, testLabels), epochs = 20, shuffle = True)
    model.save_weights(filename)


# function to freeze all but the last layer of a model and re-initialize the final layer
# input:
# model - the model to reinitialize
def reInitializeFinalLayer(model):
    print("re-initializing model")
    initModel = model

    # freeze all the layers except the last
    initModel.trainable = False
    layer = initModel.layers[-1]
    layer.trainable = True
    
    # re-initialize the last layer of the network
    weights = [layer.kernel, layer.bias]
    initializers = [layer.kernel_initializer, layer.bias_initializer]

    for w, init in zip(weights, initializers):
        w.assign(init(w.shape, dtype=w.dtype))
    



# main method
def main():
    # read dataset
    trainingDic, testDic = readData(DATASET_PATH)

    # get the training data and labels
    trainingData = trainingDic['data']
    trainingLabels = trainingDic['labels']
    # get the testing data and labels
    testData = testDic['data']
    testLabels = testDic['labels']

    # reshape the data to suitable dimensions for the model
    print("reshaping data")
    # reshape training data
    trainingData = np.transpose(np.reshape(trainingData, (-1, IMAGE_CHANNELS, IMAGE_DIM, IMAGE_DIM)), (0,2,3,1))
    # reshape the training labels
    raveled = np.ravel(trainingLabels)
    trainingLabels = np.zeros(shape=(len(raveled), NO_OF_CLASSES))
    for i, val in enumerate(raveled):
        tmp = np.zeros(shape=(NO_OF_CLASSES))
        tmp[val - 1] = 1
        trainingLabels[i] = tmp
    print("training data shape: ", np.shape(trainingData))
    print("training labels shape: ", np.shape(trainingLabels))

    # reshape the testing data
    testData = np.transpose(np.reshape(testData, (-1, IMAGE_CHANNELS, IMAGE_DIM, IMAGE_DIM)), (0,2,3,1))
    # reshape the test labels
    raveled = np.ravel(testLabels)
    testLabels = np.zeros(shape=(len(raveled), NO_OF_CLASSES))
    for i, val in enumerate(raveled):
        tmp = np.zeros(shape=(NO_OF_CLASSES))
        tmp[val - 1] = 1
        testLabels[i] = tmp
    print("test data shape: ", np.shape(testData))
    print("test labels shape: ", np.shape(testLabels))

    # save the first image for viewing to make sure the data is read correctly
    saveImage(trainingData[0])

    # make a convolutional network
    cnn = model(trainingData)

    # train the model
    # trainModel(cnn, CNN_MODEL_FILE, trainingData, trainingLabels, testData, testLabels)
    
    # load the model
    cnn.load_weights(CNN_MODEL_FILE)

    print("final layers weights before re-initialization:")
    print(cnn.layers[-1].get_weights()[0])

    # re-initialize the model so the final layer is scrambled
    reInitializeFinalLayer(cnn)

    print("final layers weights after re-initialization:")
    print(cnn.layers[-1].get_weights()[0])

    # evaluate the model before using the optimization algorithm
    print("=====================================================\nEvaluating model before using optimization algorithm\n=====================================================")
    score = cnn.evaluate(testData, testLabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # train the model using the optimization algorithm

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model after using optimization algorithm\n=====================================================")
    score = cnn.evaluate(testData, testLabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# run the main method
main()