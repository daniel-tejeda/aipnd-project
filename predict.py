#PROGRAMMER: Daniel Tejeda
from classifier import Classifier
from utils import *


def main():

    in_arg = get_predict_input_args()

    classifier = Classifier(in_arg)
    classifier.predict()


# Call to main function to run the program
if __name__ == "__main__":
    main()
