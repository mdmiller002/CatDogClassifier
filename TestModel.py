"""
This module tests the model built via BuildModel
"""

import numpy as np
import cv2
import os
import sys
import time
import Config
from keras.preprocessing import image
import argparse


def TestModel(modelIsFile, model):

    """
    Test a model
    :param modelIsFile: Boolean, if the model is a file. If it isn't,
                        then it is a Keras model object
    :param model: Model file, or object
    :return: (float, float, float, float)
             Average cat accuracy, average dog accuracy, average total accuracy, average time
    """

    from keras.models import load_model
    from keras.preprocessing import image

    testDirectory = 'data/test_set/'
    CAT_CLASS = 0
    DOG_CLASS = 1

    if modelIsFile:
        classifier = load_model(model)
    else:
        classifier = model


    # Track average times, and cat & dog accuracy
    times = []
    correctCatGuesses = 0
    numCatGuesses = 0
    correctDogGuesses = 0
    numDogGuesses = 0

    # Run inferences on all images in the testing directory
    for root, dirs, files in os.walk(testDirectory):
        for imgFile in files:
            if imgFile.endswith('.jpg'):

                # Run the inference on the image
                imgPath = os.path.join(root, imgFile)
                start = time.time()
                img = cv2.imread(imgPath)
                if Config.imgChannels == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (Config.imgRows, Config.imgCols))
                img = np.reshape(img, [1, Config.imgRows, Config.imgCols, Config.imgChannels])
                #img = image.load_img(imgPath,
                #                     target_size=(Config.imgRows, Config.imgCols),
                #                     color_mode='grayscale')
                #img = image.img_to_array(img)
                #img = np.expand_dims(img, axis=0)
                result = classifier.predict_classes(img)
                end = time.time()

                if 'cats' in imgPath:
                    if result[0][0] == CAT_CLASS:
                        correctCatGuesses += 1
                    numCatGuesses += 1

                if 'dogs' in imgPath:
                    if result[0][0] == DOG_CLASS:
                        correctDogGuesses += 1
                    numDogGuesses += 1

                times.append((end - start) * 1000)
                #print('*', end='')
                #sys.stdout.flush()
    #print()


    # Calculate average accuracies and times of the runs
    accuracy = (correctCatGuesses + correctDogGuesses) / (numCatGuesses + numDogGuesses) * 100
    catAccuracy = correctCatGuesses / numCatGuesses * 100
    dogAccuracy = correctDogGuesses / numDogGuesses * 100
    averageTime = sum(times) / len(times)

    print('Average accuracy: %3.2f%%' % accuracy)
    print('Average cat accuracy: %3.2f%%' % catAccuracy)
    print('Average dog accuracy: %3.2f%%' % dogAccuracy)
    print('Average time: %8.2fms' % averageTime)

    return catAccuracy, dogAccuracy, accuracy, averageTime



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test a model')

    parser.add_argument('model',
                        type=str,
                        help='Model file to test')

    args = parser.parse_args()

    TestModel(True, args.model)
