"""
This module tests the model built via BuildModel

If running this file standalone, the usage is:
    python.exe TestModel.py model_file'
"""

import numpy as np
import cv2
import os
import sys
import time



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
                img = cv2.resize(img, (150, 150))
                img = np.reshape(img, [1, 150, 150, 3])
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
                print('*', end='')
                sys.stdout.flush()
    print()


    # Calculate average accuracies and times of the runs
    accuracy = (correctCatGuesses + correctDogGuesses) / (numCatGuesses + numDogGuesses) * 100
    catAccuracy = correctCatGuesses / numCatGuesses * 100
    dogAccuracy = correctDogGuesses / numDogGuesses * 100
    averageTime = sum(times) / len(times)

    print('Average accuracy: %3.2f%%' % accuracy)
    print('Average cat accuracy: %3.2f%%' % catAccuracy)
    print('Average dog accuracy: %3.2f%%' % dogAccuracy)
    print('Average time: %8.2fms' % averageTime)

    return catAccuracy, dogAccuracy, accuracy, time


def main():
    usage = 'Usage: python.exe TestModel.py model_file'
    if len(sys.argv) < 2:
        print(usage)
        return

    if '-h' in sys.argv:
        print(usage)
        return

    model = sys.argv[1]
    TestModel(True, model)


if __name__ == '__main__':
    main()