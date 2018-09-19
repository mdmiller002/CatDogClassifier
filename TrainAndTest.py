"""
Train and test a model incrementally

Usage: python.exe TrainAndTest.py [model_file]
            - model_file: model file to pick up training with (optional)
"""

import BuildModel
import TestModel
import csv
import sys


def main():

    epochs = 5

    modelFile = 'newModel.h5'
    writeMode = 'w'
    newModel = True
    if len(sys.argv) > 1:
        modelFile = sys.argv[1]
        newModel = False
        writeMode = 'a'

    with open('results.csv', writeMode, newline='') as file:
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

        for i in range(2, 1000):
            print('Iteration', i)
            BuildModel.BuildModel(epochs, modelFile)
            cat, dog, acc, time = TestModel.TestModel(True, modelFile)
            writer.writerow([i * epochs, cat, dog, acc, time])
            file.flush()




if __name__ == '__main__':
    main()