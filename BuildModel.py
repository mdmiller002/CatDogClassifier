"""
This module builds a model that is a binary classifier

If running this file standalone, the usage is:
    python.exe BuildModel.py num_epochs [model_file]
        - num_epochs: number of epochs to train for
        - model_file: model file to pick up training with (optional)
"""


import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys
import Config




def BuildModel(epochs, model=None):
    """
    Build the classifier model
    :param: epochs - number of epochs to train for
    :param: model - model file, if exists, to load and continue training
    :return: Sequential model instance of the built network
    """

    from keras import models
    from keras import layers
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

    batchSize = 64
    nTrainingSamples = 6919
    nTestingSamples = 2023
    modelFile = 'newModel.h5'

    # If we don't have a working model, make a new model from scratch
    if model is None:
        print('No arguments -- creating new model from scratch and writing to', modelFile)

        # Create the keras model
        classifier = models.Sequential()
        inputShape = (Config.imgRows, Config.imgCols, Config.imgChannels)

        classifier.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
        classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

        classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))
        classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

        classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
        classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

        classifier.add(layers.Flatten())

        classifier.add(layers.Dense(units=128, activation='relu'))
        classifier.add(layers.Dropout(0.5))
        classifier.add(layers.Dense(1, activation='sigmoid'))

        classifier.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    # If we do have a model, load this model and continue training it
    else:
        modelFile = model
        print('Using model', modelFile, 'to continue training with')
        classifier = models.load_model(modelFile)

    # Augment training and tesing images
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators for training and testing data
    if Config.imgChannels == 1:
        colorMode = 'grayscale'
    else:
        colorMode = 'rgb'
    trainingGenerator = train_datagen.flow_from_directory('data/training_set',
                                                     target_size=(Config.imgRows, Config.imgCols),
                                                     batch_size=batchSize,
                                                     class_mode='binary',
                                                     color_mode=colorMode)


    testingGenerator = test_datagen.flow_from_directory('data/test_set',
                                                target_size=(Config.imgRows, Config.imgCols),
                                                batch_size=batchSize,
                                                class_mode='binary',
                                                color_mode=colorMode)

    # Save the classes to a file for use later
    with open('classes.txt', 'w') as classesFile:
        classesFile.write(str(trainingGenerator.class_indices))

    # Set up callbacks to use during training process
    tensorboardCb = TensorBoard(log_dir="logs/{}".format(time()))

    # Checkpoint callback will save the model throughout to a checkpointed file
    checkpointCb = ModelCheckpoint(modelFile,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')

    callbacksList = [checkpointCb, tensorboardCb]

    # Train the model, checkpointing along the way
    history = classifier.fit_generator(trainingGenerator,
                             steps_per_epoch=(nTrainingSamples // batchSize),
                             epochs=epochs,
                             validation_data=testingGenerator,
                             validation_steps=(nTestingSamples // batchSize),
                             callbacks=callbacksList)

    #PlotTrainingAccuracy(history)
    #PlotTrainingLosses(history)

    return classifier


def PlotTrainingAccuracy(history):
    """
    Plot training accuracy
    :param: history - return value of fit training method
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def PlotTrainingLosses(history):
    """
    Plot training losses
    :param history: return value of fit training method
    """
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def main():
    usage = 'Usage: python.exe BuildModel.py number_epochs [model_file]'
    if len(sys.argv) < 2:
        print(usage)
        return

    if '-h' in sys.argv:
        print(usage)
        return

    epochs = int(sys.argv[1])
    model = None
    if len(sys.argv) >= 3:
        model = sys.argv[2]
    BuildModel(epochs, model)

if __name__ == '__main__':
    main()
