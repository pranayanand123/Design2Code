# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:06:41 2018

@author: pranay
"""

import os
import cv2
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

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

#Creating vocabulary for text

setText = [x.split() for x in texts]
setText2 = list(set(x for l in setText for x in l))
#A dictionary mapping text or symbol to integer
tokenizer = Tokenizer(filters='', split=" ", lower=False)
#Fitting on vocabulary 
tokenizer.fit_on_texts(setText2)
#One spot for the empty word in the vocabulary 
vocab_size = len(tokenizer.word_index) + 1
# Mapping the input sentences into the vocabulary indexes
train_sequences = tokenizer.texts_to_sequences(texts)
# The longest set of design tokens
max_sequence = max(len(s) for s in train_sequences)
# No. of tokens to have in each input sentence
max_length = 48

def preprocess_data(sequences, features):
    X, y, image_data = list(), list(), list()
    for img_no, seq in enumerate(sequences):
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            # Padding the input token sentences to max_sequence with 0
            in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
            # Turning the output into one-hot encoding
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # Add the corresponding image to the boostrap token file
            image_data.append(features[img_no])
            # Limit the input sentence to 48 tokens and add it
            X.append(in_seq[-48:])
            y.append(out_seq)
    return np.array(X), np.array(y), np.array(image_data)

X, y, image_data = preprocess_data(train_sequences, train_features)


