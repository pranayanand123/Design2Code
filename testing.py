# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 02:03:42 2018

@author: pranay
"""

from os import listdir
import cv2
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import pickle
import os
from keras.models import load_model

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

tokenizer = Tokenizer(filters='', split=" ", lower=False)

with open('unique', 'rb') as f:
    unique = pickle.load(f)
    
tokenizer.fit_on_texts([' '.join(unique)])


train_features, texts = load_data(data_dir)
model = load_model('androidweights30.h5')

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    photo = np.array([photo])
    # seed the generation process
    in_text = '<START> '
    # iterate over the whole length of the sequence
    print('\nPrediction---->\n\n<START> ', end='')
    for i in range(150):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += word + ' '
        # stop if we predict the end of the sequence
        print(word + ' ', end='')
        if word == '<END>':
            break
    return in_text


max_length = 48 

for i in range(len(texts)):
    
    yhat = generate_desc(model, tokenizer, train_features[i], max_length)
    
