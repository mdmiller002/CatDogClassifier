"""
Train and test a model incrementally
"""

import BuildModel
import TestModel
import csv
import argparse


def TrainAndTest(epochs, iterations, model, resultsCsv):
	print('iterations:'+str(iterations)+'\tEpochs:'+str(epochs))
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

	with open(resultsFile, writeMode, newline='') as results_file, open("PerEpochMetrics.csv", writeMode, newline='') as PerE_file:
		results_writer = csv.writer(results_file)
		PerE_writer = csv.writer(PerE_file)
		if writeMode == 'w':
			results_writer.writerow(['Epochs', 'Cat Accuracy', 'Dog Accuracy', 'Total Accuracy', 'Average Time'])
			PerE_writer.writerow(['Epochs','val_loss','val_acc','loss','acc'])
			results_file.flush()
			PerE_file.flush()

		# Make and test one model, so we have a new model results_file
		if newModel:
			hist = BuildModel.BuildModel(epochs, None)
		else:
			hist = BuildModel.BuildModel(epochs, modelFile)
		cat, dog, acc, time = TestModel.TestModel(True, modelFile)

		# write history
		val_loss = hist.history['val_loss']
		val_acc = hist.history['val_acc']
		loss = hist.history['loss']
		acc = hist.history['acc']
		for epoch in hist.epoch:
			PerE_writer.writerow([epoch,val_loss[epoch],val_acc[epoch],loss[epoch],acc[epoch]])
		PerE_file.flush()
		results_writer.writerow([epochs, cat, dog, acc, time])
		results_file.flush()

		for i in range(1, iterations):
			print('Iteration', i)
			BuildModel.BuildModel(epochs, modelFile)
			cat, dog, acc, time = TestModel.TestModel(True, modelFile)
			results_writer.writerow([i * epochs, cat, dog, acc, time])

			# write history
			val_loss = hist.history['val_loss']
			val_acc = hist.history['val_acc']
			loss = hist.history['loss']
			acc = hist.history['acc']
			for epoch in hist.epoch:
				PerE_writer.writerow([i * epochs+epoch, val_loss[epoch], val_acc[epoch], loss[epoch], acc[epoch]])
			PerE_file.flush()
			results_file.flush()

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
