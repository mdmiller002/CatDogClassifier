"""
This module builds a model that is a binary classifier
"""


import matplotlib.pyplot as plt
import Config
import argparse
import os
from datetime import datetime



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
    from keras.callbacks import ModelCheckpoint

    batchSize = 128
    trainingPath = 'data/training_set'
    testingPath = 'data/test_set'

    nTrainingSamples = sum([len(files) for r, d, files in os.walk(trainingPath)])
    nTestingSamples = sum([len(files) for r, d, files in os.walk(testingPath)])
    now = datetime.now()
    modelFile = ''.join(['newModel_', now.strftime("%b%d_%I%M%S%p"), '.h5'])

    # If we don't have a working model, make a new model from scratch
    if model is None:
        print('No arguments -- creating new model from scratch and writing to', modelFile)

        # Create the keras model
        classifier = models.Sequential()
        inputShape = (Config.imgRows, Config.imgCols, Config.imgChannels)

        classifier.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
        classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))
        classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
        classifier.add(layers.Dropout(0.25))

        classifier.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
        classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
        classifier.add(layers.Dropout(0.2))

        classifier.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
        classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
        classifier.add(layers.Dropout(0.25))

        classifier.add(layers.Flatten())

        classifier.add(layers.Dense(units=512, activation='relu'))
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
    trainingGenerator = train_datagen.flow_from_directory(trainingPath,
                                                     target_size=(Config.imgRows, Config.imgCols),
                                                     batch_size=batchSize,
                                                     class_mode='binary',
                                                     color_mode=colorMode)


    testingGenerator = test_datagen.flow_from_directory(testingPath,
                                                target_size=(Config.imgRows, Config.imgCols),
                                                batch_size=batchSize,
                                                class_mode='binary',
                                                color_mode=colorMode)

    # Save the classes to a file for use later
    with open('classes.txt', 'w') as classesFile:
        classesFile.write(str(trainingGenerator.class_indices))

    # Set up callbacks to use during training process
    #tensorboardCb = TensorBoard(log_dir="logs/{}".format(time()))

    # Checkpoint callback will save the model throughout to a checkpointed file
    checkpointCb = ModelCheckpoint(modelFile,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=False,
                                   mode='max')

    callbacksList = [checkpointCb]

    # Train the model, checkpointing along the way
    history = classifier.fit_generator(trainingGenerator,
                             steps_per_epoch=(nTrainingSamples // batchSize),
                             epochs=epochs,
                             validation_data=testingGenerator,
                             validation_steps=(nTestingSamples // batchSize),
                             callbacks=callbacksList)

    PlotTrainingAccuracy(history)
    PlotTrainingLosses(history)

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
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a deep learning model')

    parser.add_argument('epochs',
                        type=int,
                        help='Number of epochs to train for')

    parser.add_argument('--model',
                        type=str,
                        help='Model file to continue training with')

    args = parser.parse_args()

    BuildModel(args.epochs, args.model)
