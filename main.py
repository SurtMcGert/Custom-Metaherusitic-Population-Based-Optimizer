# Computational Intelligence Course Work
# imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import classification_report
from cnn import *
from batOptimizer import batOptimizer
from geneticOptimizer import GeneticOptimizer
from greyWolfOptimizer import GreyWolfOptimizer
from rcgaOptimizer import RCGAOptimizer
from nsga_iiOptimizer import NSGAIIOptimizer
from torchvision import transforms
import matplotlib
import time
import os
matplotlib.use("Agg")

# global variables
DATASET_PATH = 'dataset'  # the directory that the dataset files are
IMAGE_DIM = 32  # the dimension of all the images is 32 x 32
IMAGE_CHANNELS = 3  # each image is RGB so has 3 channels
NO_OF_CLASSES = 10  # there are 10 classes of images in the dataset
# the name of the files for storing trained networks and training history
CNN_MODEL_FILE = "cnnModel"
CNN_MODEL_TRAIN_HISTORY_FILE = "cnnModelHistory"
MODEL_WITH_ALGORITHM_FILE = "cnnWithAlgorithm"
MODEL_WITH_ALGORITHM_TRAIN_HISTORY_FILE = "cnnWithAlgorithmHistory"
BCGA_MODEL_FILE = "bcgaModel"
BCGA_MODEL_TRAIN_HISTORY_FILE = "bcgaModelHistory"
RCGA_MODEL_FILE = "rcgaModel"
RCGA_MODEL_TRAIN_HISTORY_FILE = "rcgaModelHistory"
BAT_MODEL_FILE = "batModel"
BAT_MODEL_TRAIN_HISTORY_FILE = "batModelHistory"
WOLF_MODEL_FILE = "wolfModel"
WOLF_MODEL_TRAIN_HISTORY_FILE = "wolfModelHistory"
NSGAII_MODEL_FILE = "nsgaiiModel"
NSGAII_MODEL_TRAIN_HISTORY_FILE = "NSGAIIModelHistory"
# define training hyperparameters
BATCH_SIZE = 128
EPOCHS = 9
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT


# function to build a CNN
# inputs:
# device - the device to train the model on
# trainingData - the data that will be used to train the model
# channels - the number of channels the images have
#
# returns: a CNN model, the optimizer and the loss function
def modelCNN(device, trainingData, channels):
    print("making a CNN")
    print("device: ", device)
    model = CNN(
        numChannels=channels,
        classes=len(trainingData.dataset.classes)).to(device)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=0.001)
    lossFn = nn.NLLLoss()
    return model, opt, lossFn


# function to train a given model and save its weights to a file
# inputs:
# device - the device to train the model on
# device - the device to train the model on
# model - the model to train
# opt - the optimization algorithm
# lossFn - the loss function
# trainingDataLoader - the data loader for the training data
# valDataLoader - the data loader for the validation data
# epochs - the number of epochs
# batchSize - the batch size
#
# returns: the trained model and the training history
def trainModel(device, model, opt, lossFn, trainingDataLoader, valDataLoader, epochs, batchSize):
    print("training model")
    model = model.to(device)
    trainStart = time.time()
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainingDataLoader.dataset) // batchSize
    valSteps = len(valDataLoader.dataset) // batchSize
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # loop over our epochs
    for e in range(0, epochs):
        epochStart = time.time()
        # set the model in training mode
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in trainingDataLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)
            model.y = y
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in valDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

            epochEnd = time.time()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainingDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
            avgValLoss, valCorrect))
        epochTimeTaken = (epochEnd - epochStart) / 60
        print("time to train epoch: ", epochTimeTaken, " minuets\n")

    trainEnd = time.time()
    trainTimeTaken = (trainEnd - trainStart) / 60
    print("time to train: ", trainTimeTaken, " minuets\n")
    return model, H


# function to save a model and its history
# inputs:
# model - the model to save
# H - the training history for the model
# modelFileName - the file name to save the model into
# historyFileName - the file name to save the training history into
def saveModel(model, H, modelFileName, historyFileName):
    print("saving model: ", modelFileName)
    folder = "models/"
    torch.save(model, folder+modelFileName)
    torch.save(H, folder+historyFileName)


# function to load a model and its history
# inputs:
# modelFileName - the file name to load the model from
# historyFileName - the file name to load the training history from
#
# returns: the model, and the history
def loadModel(modelFileName, historyFileName):
    print("loading model: ", modelFileName)
    folder = "models/"
    model = torch.load(folder+modelFileName)
    H = torch.load(folder+historyFileName)
    return model, H


# function to save a model and its history
# inputs:
# model - the model to save
# H - the training history for the model
# modelFileName - the file name to save the model into
# historyFileName - the file name to save the training history into
def saveModel(model, H, modelFileName, historyFileName):
    print("saving model: ", modelFileName)
    folder = "models/"
    torch.save(model, folder+modelFileName)
    torch.save(H, folder+historyFileName)


# function to load a model and its history
# inputs:
# modelFileName - the file name to load the model from
# historyFileName - the file name to load the training history from
#
# returns: the model, and the history
def loadModel(modelFileName, historyFileName):
    print("loading model: ", modelFileName)
    folder = "models/"
    model = torch.load(folder+modelFileName)
    H = torch.load(folder+historyFileName)
    return model, H

# function to check if the model file exists
# inputs:
# file - the file name to look for
#
# returns: wether the file exists or not


def trainingFileExists(file):
    folder = "models/"
    return os.path.isfile(folder + file)

# function to evaluate a model
# inputs:
# device - the device this model was trained on
# model - the model to evaluate
# testDataLoader - the data loader for the test data
# testingData - the testing data
# H - the training history
# plotName - the name of the file to save the plot for this evaluation


def evaluateModel(device, model, testDataLoader, testingData, H, plotName):
    print("evaluating model...")
    folder = "evaluations/"
    model = model.to(device)
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
    # generate a classification report
    print(classification_report(testingData.targets,
                                np.array(preds), target_names=testingData.classes))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(folder+plotName)


# main method
def main():
    print("getting training and testing data")
    trainingData = CIFAR10(root="dataset", train=True, download=True,
                           transform=ToTensor())
    testingData = CIFAR10(root="dataset", train=False, download=True,
                          transform=ToTensor())
    # calculate the train/validation split
    print("generating the train/validation split...")
    numTrainSamples = int(len(trainingData) * TRAIN_SPLIT)
    numValSamples = int(len(trainingData) * VAL_SPLIT)
    (trainingData, valData) = random_split(trainingData,
                                           [numTrainSamples, numValSamples],
                                           generator=torch.Generator().manual_seed(42))

    # initialize the train, validation, and test data loaders
    print("initializing the data loaders...")
    trainingDataLoader = DataLoader(trainingData, shuffle=True,
                                    batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(testingData, batch_size=BATCH_SIZE)

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make a CNN model
    cnn, opt, lossFn = modelCNN(device, trainingData, IMAGE_CHANNELS)

    # train the resnet model
    if trainingFileExists(CNN_MODEL_FILE):
        cnn, H = loadModel(
            CNN_MODEL_FILE, CNN_MODEL_TRAIN_HISTORY_FILE)
    else:
        cnn, H = trainModel(
            device, cnn, opt, lossFn, trainingDataLoader, valDataLoader, EPOCHS, BATCH_SIZE)
        # reset the last layer of the resnet model
        cnn.reInitializeFinalLayer()
        # save resnet model to disk
        saveModel(cnn, H, CNN_MODEL_FILE, CNN_MODEL_TRAIN_HISTORY_FILE)

    # evaluate the model before using the optimization algorithm
    print("=====================================================\nEvaluating resnet model before using optimization algorithms\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "originalResnetEvaluationPlot.png")

    # train the model using the genetic optimization algorithm
    print("binary coded genetic optimizer")
    opt = GeneticOptimizer(device, cnn, lossFn=lossFn,
                           weightLowerBound=-1, weightUpperBound=1, numOfBits=8, pop=20, elites=10)
    if trainingFileExists(BCGA_MODEL_FILE):
        cnn, H = loadModel(
            BCGA_MODEL_FILE, BCGA_MODEL_TRAIN_HISTORY_FILE)
    else:
        cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
                            valDataLoader, EPOCHS, BATCH_SIZE)
        saveModel(cnn, H, BCGA_MODEL_FILE, BCGA_MODEL_TRAIN_HISTORY_FILE)

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model after using binary coded genetic algorithm\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "geneticAlgorithmEvaluationPlot.png")
    cnn.reInitializeFinalLayer()

    # train the model using the real coded genetic optimization algorithm
    print("real coded genetic optimizer")
    opt = RCGAOptimizer(device, cnn, lossFn=lossFn,
                        weightLowerBound=-1, weightUpperBound=1, pop=40, elites=10)
    if trainingFileExists(RCGA_MODEL_FILE):
        cnn, H = loadModel(
            RCGA_MODEL_FILE, RCGA_MODEL_TRAIN_HISTORY_FILE)
    else:
        cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
                            valDataLoader, EPOCHS, BATCH_SIZE)
        saveModel(cnn, H, RCGA_MODEL_FILE, RCGA_MODEL_TRAIN_HISTORY_FILE)

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model after using real coded genetic algorithm\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "RCGAEvaluationPlot.png")
    cnn.reInitializeFinalLayer()

    # # train the model using the grey wolf optimization algorithm
    if trainingFileExists(WOLF_MODEL_FILE):
        cnn, H = loadModel(
            WOLF_MODEL_FILE, WOLF_MODEL_TRAIN_HISTORY_FILE)
    else:
        opt = GreyWolfOptimizer(device, cnn, lossFn,
                                numOfIters=len(trainingDataLoader.dataset), pop=100, debug=False)
        cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
                            valDataLoader, EPOCHS, BATCH_SIZE)
        saveModel(cnn, H, WOLF_MODEL_FILE, WOLF_MODEL_TRAIN_HISTORY_FILE)

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model after using grey wolf algorithm\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "greyWolfAlgorithmEvaluationPlot.png")
    cnn.reInitializeFinalLayer()

    # train the model using the bat optimizer
    print("bat PS optimizer")
    if trainingFileExists(BAT_MODEL_FILE):
        cnn, H = loadModel(
            BAT_MODEL_FILE, BAT_MODEL_TRAIN_HISTORY_FILE)
    else:
        opt = batOptimizer(device, cnn, lossFn,
                           populationSize=10, max_iters=20)
        cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
                            valDataLoader, EPOCHS, BATCH_SIZE)
        saveModel(cnn, H, BAT_MODEL_FILE, BAT_MODEL_TRAIN_HISTORY_FILE)

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model model after using bat algorithm\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "originalResNetBatAlgorithmEvaluationPlot.png")
    cnn.reInitializeFinalLayer()

    # evaluate the model after using the NSGAII optimization algorithm
    # train the model using the grey wolf optimization algorithm
    if trainingFileExists(NSGAII_MODEL_FILE):
        cnn, H = loadModel(
            NSGAII_MODEL_FILE, NSGAII_MODEL_TRAIN_HISTORY_FILE)
    else:
        opt = GreyWolfOptimizer(device, cnn, lossFn, pop=10, max_iters=20)
        cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
                            valDataLoader, EPOCHS, BATCH_SIZE)

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model after using NSGAII algorithm\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "NSGAIIAlgorithmEvaluationPlot.png")
    cnn.reInitializeFinalLayer()


# run the main method
main()
