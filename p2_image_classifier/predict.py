#!/usr/bin/env python

import numpy as np
from PIL import Image

# Disable CUDA messages:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import tensorflow_hub as hub
import json
import sys
import getopt
import signal


def predict(image_path, model, top_k):
    ''' Predicts the flower category for the given image_path,
        using the given machine learning model, and returns the
        top_k probabilities and classes.
        
        Note that the classes are adjusted to account for the 
        1-based vs 0-based indexing in the class names JSON file.
    '''
    image = process_image(np.asarray(Image.open(image_path)))
    image = np.expand_dims(image, axis=0)
    
    pred = model.predict(image)
    
    pred = pred[0].tolist()
    
    probs, classes = tf.math.top_k(pred, k=top_k)
    
    return probs, classes + 1 # Correct class index to 1-based used in class_names

def process_image(np_image):
    ''' Takes in an image that consists of numpy arrays,
        resizes to 224 x 224 pixels, and normalizes the 
        color values to 0-1. 
        
        Returns image as set of numpy arrays.
    '''
    image_size = 224
    tf_image = tf.convert_to_tensor(np_image)
    tf_image = tf.image.resize(np_image, [image_size, image_size])
    tf_image /= 255.0
    return tf_image.numpy()


def sigint_handler(signum, frame):
    ''' Catches interrupts such as Ctrl-C and exits nicely instead
        of throwing up an unnecessary stack trace. Unfortunately it
        does not catch a SIGINT event during imports.
    '''
    print('Caught interrupt; exiting nicely.')
    exit(0)
    
def print_help():
    print('Usage: python predict.py <image path> <model path>')
    print('\nOptional arguments:')
    print('\t--top_k <K>\t\t\t\tNumber of probabilities and classes to return')
    print('\t--category_names <JSON name file path>\tName of JSON file containing category to index mapping')
    print('\t-p\t\t\t\t\tPlot image and prediction results')

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    
    # Make sure the input image path and model name are not missing.
    if len(sys.argv) < 3:
        print('Not enough arguments.\n')
        print_help()
        exit(0)
        
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Set default values for optional arguments
    top_k = 5
    cat_names_path = './label_map.json'
    plot_flag = False
    
    # Modify values for optional arguments if specified on command line
    opts, args = getopt.gnu_getopt(sys.argv[3:], 't:c:p', ['top_k=', 'category_names=', 'plot'])
    for opt, arg in opts:
        if opt in ('-t', '--top_k'):
            top_k = int(arg)
        elif opt in ('-c', '--category_names'):
            cat_names_path = arg
        elif opt in ('-p', '--plot'):
            plot_flag = True
        else:
            print('Unknown argument.')
            print_help()
        
    with open(cat_names_path, 'r') as f:
        class_names = json.load(f)

    # Load model and make predictions
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    probs, classes = predict(image_path, model, top_k)
    
    print(f'\nThe flower was predicted to be: {class_names[str(classes.numpy()[0])]}, with {round(probs.numpy()[0]*100, 2)} % probability.')
    
    categories = []
    for flower_class in classes.numpy():
        categories.append(class_names[str(flower_class)])
    
    print(f'\nThe top {top_k} results:')
    print('===================')
    for i in range(len(probs.numpy())):
        # Some kludging because the rounded probability isn't working in formatted string
        prob = np.round(probs.numpy()[i], 4)
        print('Probability: ', end='')
        print(prob, end='')
        print(f',\tCategory: {classes.numpy()[i]},\tName: {categories[i].title()}')
    
    if plot_flag:
        # Only import if actually plotting
        import matplotlib.pyplot as plt
        
        
    
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(Image.open(image_path))
        # ax1.set_title('Input Test Image')
        ax2.barh(categories, probs) #sorted(probs.numpy().tolist()))
        ax2.invert_yaxis()
        ax2.set_title('Class Probability')
        ax2.set_xlim([0.0, 1.0])
        plt.tight_layout()
        plt.show()    
    
if __name__ == '__main__':
    main()