"""
Train and test a model incrementally
"""

import csv
import argparse
from _datetime import datetime
import os

def TrainAndTest(epochs, iterations, model, resultsCsv,timeStamp = None):
    import BuildModel
    import TestModel

    print('iterations:' + str(iterations) + '\tEpochs:' + str(epochs))
    now = datetime.now()
    if timeStamp==None:
        timeStamp = now.strftime("%b%d_%I%M%S%p")

    modelFile = ''.join(['newModel_', timeStamp])
    save_root = 'model/'+modelFile+'/'
    modelFile = modelFile+'.h5'
    writeMode = 'w'
    newModel = True

    if model is not None:
        save_root = model.split('/')
        if(len(save_root)!=1):
            modelFile = save_root[-1]
            save_root = '/'.join(save_root[:-1])+'/'

        else:
            save_root = ''
            modelFile = model
        newModel = False
        writeMode = 'a'

    resultsFile = 'results.csv'
    if resultsCsv is not None:
        resultsFile = resultsCsv
    if not os.path.isdir(save_root) and newModel:
        os.makedirs(save_root)

    with open(save_root+resultsFile, writeMode, newline='') as results_file,\
            open(save_root+"PerEpochMetrics.csv", writeMode,newline='') as PerE_file:

        results_writer = csv.writer(results_file)
        PerE_writer = csv.writer(PerE_file)
        if writeMode == 'w':
            results_writer.writerow(['Epochs', 'Cat Accuracy', 'Dog Accuracy', 'Total Accuracy', 'Average Time'])
            PerE_writer.writerow(['Epochs', 'val_loss', 'val_acc', 'loss', 'acc'])
            results_file.flush()
            PerE_file.flush()

        # Make and test one model, so we have a new model results_file

        for i in range(0, iterations):
            print('Iteration', i+1)

            if newModel and i == 0:
                hist, modelFile = BuildModel.BuildModel(epochs, None,timeStamp=timeStamp)
            else:
                hist, modelFile = BuildModel.BuildModel(epochs, save_root+modelFile)

            cat, dog, acc, time = TestModel.TestModel(True, save_root+modelFile)
            results_writer.writerow([i * epochs, cat, dog, acc, time])

            # write history
            val_loss = hist.history['val_loss']
            val_acc = hist.history['val_acc']
            loss = hist.history['loss']
            acc = hist.history['acc']
            for epoch in hist.epoch:
                PerE_writer.writerow([i * epochs + epoch, val_loss[epoch], val_acc[epoch], loss[epoch], acc[epoch]])
            PerE_file.flush()
            results_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model and test along the way')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to train during each iteration')

    parser.add_argument('--iterations',
                        type=int,
                        default=100,
                        help='Number of iterations to train for the '
                             'number of epochs and test')

    parser.add_argument('--model',
                        type=str,
                        help='Model file to continue training with')

    parser.add_argument('--outputCsv',
                        type=str,
                        help='CSV file to output file to')

    parser.add_argument('--timeStamp',
                        type=str,
                        help='secondary name for model')

    args = parser.parse_args()

    TrainAndTest(args.epochs, args.iterations, args.model, args.outputCsv, timeStamp=args.timeStamp)
