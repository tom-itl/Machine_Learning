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
    image = process_image(np.asarray(Image.open(image_path)))
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    
    pred = model.predict(image)
    
    pred = pred[0].tolist()
    
    probs, classes = tf.math.top_k(pred, k=top_k)
    
    return probs, classes + 1 # Correct class index to 1-based used in class_names

def process_image(np_image):
    image_size = 224
    tf_image = tf.convert_to_tensor(np_image)
#     tf_image = tf.cast(tf_image, tf.float32)
    tf_image = tf.image.resize(np_image, [image_size, image_size])
    tf_image /= 255.0
    return tf_image.numpy()


def sigint_handler(signum, frame):
    print('Caught interrupt; exiting nicely.')
    exit(0)
    
def main():
    signal.signal(signal.SIGINT, sigint_handler)
    
    if len(sys.argv) < 3:
        print('Insufficient information.') # TODO: Pad this out
        exit(0)
        
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    top_k = 5
    cat_names_path = './label_map.json'
    plot_flag = False
    
    opts, args = getopt.gnu_getopt(sys.argv[3:], 't:c:p', ['top_k=', 'category_names=', 'plot'])
    for opt, arg in opts:
        if opt in ('-t', '--top_k'):
            top_k = int(arg)
        elif opt in ('-c', '--category_names'):
            cat_names_path = arg
        elif opt in ('-p', '--plot'):
            plot_flag = True
        else:
            print('Unknown argument. Usage:') # TODO: Print usage instructions
        
    with open(cat_names_path, 'r') as f:
        class_names = json.load(f)

#     model_path = './flower_recog_model.h5'
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(image_path, model_path)
    print(model.summary)
    print(cat_names_path, top_k)
    
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)
    
    print(f'The flower was predicted to be: {class_names[str(classes.numpy()[0])]}, with {round(probs.numpy()[0]*100, 2)} % probability.')
    
    if plot_flag:
        import matplotlib.pyplot as plt
        
        categories = []
        for flower_class in classes.numpy():
            categories.append(class_names[str(flower_class)])
    
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)
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