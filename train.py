#PROGRAMMER: Daniel Tejeda
from classifier import Classifier
from utils import *


def main():

    in_arg = get_train_input_args()

    classifier = Classifier(in_arg)

    print('\nClassifier created for training with parameters:\n')
    classifier.print_args()

    classifier.train()


# Call to main function to run the program
if __name__ == "__main__":
    main()
