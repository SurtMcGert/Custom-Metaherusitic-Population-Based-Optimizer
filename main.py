# Computational Intelligence Course Work
# imports
import pickle
import os
import numpy as np 
from PIL import Image as im 

# global variables
DATASET_PATH = 'dataset' # the directory that the dataset files are
IMAGE_DIM = 32 # the dimension of all the images is 32 x 32
IMAGE_CHANNELS = 3 # each image is RGB so has 3 channels

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
            trainingData.update(unpickle(path))
        # otherwise if it starts with "test" then it is the test data
        elif filename.startswith('test'):
            testData.update(unpickle(path))
        index += 1
    return trainingData, testData


# function to save an image for viewing
# inputs:
# imageArr - the 1D array of pixel data containing data in row major order
# dimension - the dimension of the image, assuming the image is square
# channels - the number of channels the image has
def saveImage(imageArr, dimension, channels):
    print("saving image")
    # reshape the array to the correct dimension
    imageArr = np.transpose(np.reshape(imageArr, (channels, dimension, dimension)), (1,2,0))
    # show the shape of the array 
    print(imageArr.shape)
    #create an image object using the array
    data = im.fromarray(imageArr)
    # save the output as a PNG file
    data.save('image.png')


# main method
def main():
    # read dataset
    trainingDic, testDic = readData(DATASET_PATH)

    # get the training data and labels
    trainingData = trainingDic['data'.encode("UTF-8")]
    trainingLabels = trainingDic['labels'.encode("UTF-8")]
    # get the testing data and labels
    testData = testDic['data'.encode("UTF-8")]
    testLabels = testDic['labels'.encode("UTF-8")]

    # save the first image for viewing to make sure the data is read correctly
    saveImage(trainingData[0], IMAGE_DIM, IMAGE_CHANNELS)


# run the main method
main()