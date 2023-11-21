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
from cnn import CNN
from geneticOptimizer import GeneticOptimizer
import matplotlib
matplotlib.use("Agg")

# global variables
DATASET_PATH = 'dataset'  # the directory that the dataset files are
IMAGE_DIM = 32  # the dimension of all the images is 32 x 32
IMAGE_CHANNELS = 3  # each image is RGB so has 3 channels
NO_OF_CLASSES = 10  # there are 10 classes of images in the dataset
# the name of the file storing the weights of the CNN model
CNN_MODEL_FILE = "cnnModel"
# the name of the file storing the training history of the CNN
CNN_MODEL_TRAIN_HISTORY_FILE = "cnnModelHistory"
# the name of the file storing the weights of the model after using our optimization algorithm
MODEL_WITH_ALGORITHM_FILE = "cnnWithAlgorithm"
# the name of the file storing the training history for the model after using our optimization algorithm
MODEL_WITH_ALGORITHM_TRAIN_HISTORY_FILE = "cnnWithAlgorithmHistory"
# define training hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
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
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))
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

    # make a convolutional network
    cnn, opt, lossFn = modelCNN(device, trainingData, IMAGE_CHANNELS)

    # train the CNN model
    # cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
    #                     valDataLoader, EPOCHS, BATCH_SIZE)

    # reset the last layer of the model
    # cnn.reInitializeFinalLayer()

    # save the CNN model to disk
    # saveModel(cnn, H, CNN_MODEL_FILE, CNN_MODEL_TRAIN_HISTORY_FILE)
    # load the CNN model from disk
    cnn, H = loadModel(CNN_MODEL_FILE, CNN_MODEL_TRAIN_HISTORY_FILE)

    # evaluate the model before using the optimization algorithm
    print("=====================================================\nEvaluating model before using optimization algorithms\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "originalCNNEvaluationPlot.png")

    # train the model using the genetic optimization algorithm
    opt = GeneticOptimizer(device, cnn, lossFn=lossFn, pop=2, elites=1)
    opt.train(trainingDataLoader)
    # cnn, H = trainModel(device, cnn, opt, lossFn, trainingDataLoader,
    #                     valDataLoader, EPOCHS, BATCH_SIZE)

    # evaluate the model after using the optimization algorithm
    print("=====================================================\nEvaluating model after using genetic algorithm\n=====================================================")
    evaluateModel(device, cnn, testDataLoader, testingData,
                  H, "geneticAlgorithmEvaluationPlot.png")


# run the main method
main()