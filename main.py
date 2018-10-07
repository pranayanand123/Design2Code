# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:06:41 2018

@author: pranay
"""

import os
import cv2
import numpy as np

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

data_dir = 'Dataset/'

def load_data(data_dir):
    text = []
    images = []
    
    all_filenames = os.listdir(data_dir)
    for filename in (all_filenames):
        if filename[-3:] == "png":
                 
            image = cv2.imread(data_dir+filename)
            image = cv2.resize(image, (256,256))
            image= np.array(image, dtype=float)
            image = image/image.max()
            
            images.append(image)
        else:
            # Load the corresponding android tags and wrap around with start and end tag
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            # Seperate all words with a single space
            syntax = syntax.split()
            syntax = ' '.join(syntax)
            # Add a space before each comma
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=float)
    return images, text

train_features, texts = load_data(data_dir)