#PROGRAMMER: Daniel Tejeda
from classifier import Classifier
from utils import *


def main():

    in_arg = get_input_args()

    classifier = Classifier(in_arg)

    print('\nClassifier for inference with parameters:\n')
    classifier.print_args()
    
    pred, classes = classifier.predict(in_arg.img_path)
    
    pred_labels = [classifier.cat_to_name[classifier.model.idx_to_class[x.item()]] 
                                                             for x in classes[0] ]
    
    print('\n---------------Results------------------\n')
    
    for i in range(len(pred_labels)):
        print('{:>16}: {:>8.2f}%'.format(pred_labels[i], pred.data[0][i]*100))
              
    print('\n----------------------------------------\n')
    
def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('img_path', type=str, metavar='IMG_PATH',
                    help='path to image to classify')

    parser.add_argument('checkpoint', type=str, metavar='CHECKPOINT',
                    help='path to checkpoint file')
    
    parser.add_argument('--top_k', type=int, default=1,
                    help='Top K')
    
    parser.add_argument('--category_names', type=str, default='./cat_to_name.json',
                    help='path to the folder to save checkpoints')

    parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Use GPU')

    parser.set_defaults(gpu=False)

    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()
