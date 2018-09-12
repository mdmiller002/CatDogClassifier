import BuildModel
import TestModel


def main():
    model = BuildModel.BuildModel(3, None)
    TestModel.TestModel(False, model)



if __name__ == '__main__':
    main()