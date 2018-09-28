"""
Train and test a model incrementally
"""

import BuildModel
import TestModel
import csv
import argparse


def TrainAndTest(epochs, iterations, model, resultsCsv):


    modelFile = 'newModel.h5'
    writeMode = 'w'
    newModel = True
    if model is not None:
        modelFile = model
        newModel = False
        writeMode = 'a'

    resultsFile = 'results.csv'
    if resultsCsv is not None:
        resultsFile = resultsCsv


    with open(resultsFile, writeMode, newline='') as file:
        writer = csv.writer(file)
        if writeMode == 'w':
            writer.writerow(['Epochs', 'Cat Accuracy', 'Dog Accuracy', 'Total Accuracy', 'Average Time'])
            file.flush()

        # Make and test one model, so we have a new model file
        if newModel:
            BuildModel.BuildModel(epochs, None)
        else:
            BuildModel.BuildModel(epochs, modelFile)
        cat, dog, acc, time = TestModel.TestModel(True, modelFile)
        writer.writerow([epochs, cat, dog, acc, time])
        file.flush()

        for i in range(2, iterations):
            print('Iteration', i)
            BuildModel.BuildModel(epochs, modelFile)
            cat, dog, acc, time = TestModel.TestModel(True, modelFile)
            writer.writerow([i * epochs, cat, dog, acc, time])
            file.flush()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a model and test along the way')

    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to train during each iteration')

    parser.add_argument('--iterations',
                        type=int,
                        default=500,
                        help='Number of iterations to train for the '
                             'number of epochs and test')

    parser.add_argument('--model',
                        type=str,
                        help='Model file to continue training with')

    parser.add_argument('--outputCsv',
                        type=str,
                        help='CSV file to output file to')

    args = parser.parse_args()

    TrainAndTest(args.epochs, args.iterations, args.model, args.outputCsv)
