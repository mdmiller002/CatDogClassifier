"""
Train and test a model incrementally

Usage: python.exe TrainAndTest.py [model_file] [results_file]
            - model_file: model file to pick up training with (optional)
            - results_file: output CSV file to write output to (optional)
"""

import BuildModel
import TestModel
import csv
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='Train a model and test along the way')
    parser.add_argument('--epochs',
                        metavar='epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to train during each iteration')
    parser.add_argument('--model',
                        metavar='model',
                        type=str,
                        help='Model file to continue training with')
    parser.add_argument('--outputCsv',
                        metavar='output_csv',
                        type=str,
                        help='CSV file to output file to')
    args = parser.parse_args()
    print(args)

    modelFile = 'newModel.h5'
    writeMode = 'w'
    newModel = True
    if args.model is not None:
        modelFile = args.model
        newModel = False
        writeMode = 'a'

    resultsFile = 'results.csv'
    if args.outputCsv is not None:
        resultsFile = args.outputCsv


    with open(resultsFile, writeMode, newline='') as file:
        writer = csv.writer(file)
        if writeMode == 'w':
            writer.writerow(['Epochs', 'Cat Accuracy', 'Dog Accuracy', 'Total Accuracy', 'Average Time'])
            file.flush()

        # Make and test one model, so we have a new model file
        if newModel:
            BuildModel.BuildModel(args.epochs, None)
        else:
            BuildModel.BuildModel(args.epochs, modelFile)
        cat, dog, acc, time = TestModel.TestModel(True, modelFile)
        writer.writerow([args.epochs, cat, dog, acc, time])
        file.flush()

        for i in range(2, 1000):
            print('Iteration', i)
            BuildModel.BuildModel(args.epochs, modelFile)
            cat, dog, acc, time = TestModel.TestModel(True, modelFile)
            writer.writerow([i * args.epochs, cat, dog, acc, time])
            file.flush()




if __name__ == '__main__':
    main()