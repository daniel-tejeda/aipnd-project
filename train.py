#PROGRAMMER: Daniel Tejeda
from classifier import Classifier
from utils import *


def main():

    in_arg = get_input_args()

    classifier = Classifier(in_arg)

    print('\nClassifier for training with parameters:\n')
    classifier.print_args()

    classifier.train()
    
    if in_arg.validate:
        classifier.validate()

    
def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, metavar='DATA_DIR',
                    help='path to the images folder')

    parser.add_argument('--save_dir', type=str, default='./',
                    help='path to the folder to save checkpoints')

    parser.add_argument('--arch', type=str, default='vgg16',
                    help='CNN base model - options: vgg13, vgg16, alexnet')

    parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='Learning rate')

    parser.add_argument('--hidden_units', type=int, default=1024,
                    help='Number of units in the hidden layer of classifier')

    parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs for training')

    parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Use GPU')
    
    parser.add_argument('--validate', dest='validate', action='store_true',
                    help='Validate after training')
    
    parser.add_argument('--resume', dest='resume', action='store_true',
                    help='Resume training from checkpoint')

    parser.set_defaults(gpu=False)
    parser.set_defaults(validate=False)
    parser.set_defaults(resume=False)

    return parser.parse_args()


# Call to main function to run the program
if __name__ == "__main__":
    main()
