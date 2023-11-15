import torch
import torchvision.transforms
import pickle
import os
import numpy as np


class Cifar(torch.utils.data.Dataset):
    # function to load a data file into a dictionary containing:
    # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    # inputs:
    # file - path to the file to read
    #
    # returns the file as a dictionary

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # function to read the dataset
    # inputs:
    # folder - the folder that the dataset files are in
    #
    # returns a dictionary with all the data loaded

    def readData(self, folder, imageChannels, imageDim, numOfClasses, train=True):
        print("reading data")
        files = os.listdir(folder)
        index = 0
        data = {}
        # loop over all the files in the folder
        while index < len(files):
            filename = files[index]
            path = folder + "/" + filename
            # if the file name starts with "data" then it is a training data batch
            if (filename.startswith('data') & (train == True)):
                print("read ", filename)
                dic = self.unpickle(path)
                for key, value in dic.items():
                    try:
                        data[key.decode("UTF-8")].append(value)
                    except:
                        data[key.decode("UTF-8")] = [value]

            # otherwise if it starts with "test" then it is the test data
            elif (filename.startswith('test') & (train == False)):
                print("read ", filename)
                dic = self.unpickle(path)
                for key, value in dic.items():
                    try:
                        data[key.decode("UTF-8")].append(value)
                    except:
                        data[key.decode("UTF-8")] = [value]
            index += 1

        # reshape the data to suitable dimensions for the model
        print("reshaping data")
        # reshape training data
        data['data'] = np.transpose(np.reshape(
            data['data'], (-1, imageChannels, imageDim, imageDim)), (0, 2, 3, 1))

        # reshape the training labels
        data['labels'] = np.ravel(data['labels'])
        # raveled = np.ravel(trainingLabels)
        # trainingLabels = np.zeros(shape=(len(raveled), numOfClasses))
        # for i, val in enumerate(raveled):
        #     tmp = np.zeros(shape=(numOfClasses))
        #     tmp[val - 1] = 1
        #     trainingLabels[i] = tmp
        return data

    # the init function
    def __init__(self, folder, imageChannels, imageDim, numOfClasses, train=True):
        # load the dataset
        self.dataset = self.readData(
            folder, imageChannels, imageDim, numOfClasses, train)
        self.classes = np.unique(self.dataset['labels'])

    def __getitem__(self, idx):
        data, label = (self.dataset['data'])[
            idx], (self.dataset['labels'])[idx]

        transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.ToTensor()
        ])
        return transform(data), torch.tensor(label)

    def __len__(self):
        return len(self.dataset['labels'])
