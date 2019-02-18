# PROGRAMMER: Daniel Tejeda

import argparse
import sys
from PIL import Image
import numpy as np 
import torch


def process_image(image_path):

    img = Image.open(image_path)

    shortdim = (0 if img.size[0] < img.size[1] else 1)

    newsize = [*img.size]
    newsize[shortdim] = 256
    ratio = (newsize[shortdim]/float(img.size[shortdim]))
    newsize[int(not(shortdim))] = int((float(img.size[int(not(shortdim))])*float(ratio)))

    img = img.resize(newsize)

    ccsize = 224

    left = (newsize[0] - ccsize) / 2
    top = (newsize[1] - ccsize) / 2
    right = (newsize[0] + ccsize) / 2
    bottom = (newsize[1] + ccsize) / 2

    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    img = img - np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = img / np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


    return torch.Tensor(img)




def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
