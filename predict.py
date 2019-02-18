#PROGRAMMER: Daniel Tejeda
from classifier import Classifier
from utils import *


def main():

    in_arg = get_predict_input_args()

    classifier = Classifier(in_arg)

    print('\nClassifier created for inference with parameters:\n')
    classifier.print_args()
    
    pred, classes = classifier.predict(in_arg.img_path)
    
    print(pred, classes)

# Call to main function to run the program
if __name__ == "__main__":
    main()
