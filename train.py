#PROGRAMMER: Daniel Tejeda

import torch
from classifier import Classifier
from utils import *

 
def main():

    in_arg = get_train_input_args()
    print(in_arg)

    classifier = Classifier(in_arg)
    classifier.train()


# Call to main function to run the program
if __name__ == "__main__":
    main()
