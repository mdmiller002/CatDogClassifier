from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import os
import time

def main():

    # Load the model from saved files
    with open('classifier.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    classifier = model_from_json(loaded_model_json)
    classifier.load_weights('classifier_weights.h5')
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    results = []
    times = []
    path = 'data/test_set/cats/'
    for imgFile in os.listdir(path):
        if imgFile.endswith('.jpg'):

            imgPath = os.path.join(path, imgFile)
            start = time.time()
            test_image = image.load_img(imgPath, target_size=(64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict_classes(test_image)
            end = time.time()

            times.append((end - start) * 1000)
            results.append(result[0][0])

    correctGuesses = 0
    for result in results:
        if result == 1:
            correctGuesses += 1
    accuracy = correctGuesses / len(results) * 100
    print('Accuracy: %3.2f%%' % accuracy)

    averageTime = sum(times) / len(times)
    print('Time: %8.2fms' %averageTime)


if __name__ == '__main__':
    main()