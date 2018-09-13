import BuildModel
import TestModel
import csv


def main():

    epochs = 5

    with open('results.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Make and test one model, so we have a new model file
        BuildModel.BuildModel(epochs, None)
        cat, dog, acc, time = TestModel.TestModel(True, 'newModel.h5')
        writer.writerow([epochs, cat, dog, acc, time])

        for i in range(2, 1000):
            print('Iteration', i)
            BuildModel.BuildModel(epochs, 'newModel.h5')
            cat, dog, acc, time = TestModel.TestModel(True, 'newModel.h5')
            writer.writerow([i * epochs, cat, dog, acc, time])




if __name__ == '__main__':
    main()