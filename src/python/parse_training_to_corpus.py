from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence, text
from keras.models import Model, Sequential, model_from_yaml
import string
import pandas as pd
import datetime
import json
import re
from collections import defaultdict

#########################
# Write corpus to file for training word embeddings
#########################

# file_path = file to load
# load raw data
def load_word_data(file_path):
    df = pd.concat(file_gen(file_path))
    records = [line for line in df['tweet'].values]
    return records

# test
# corpus_as_array = load_word_data(TEST_DATASET)

def clean_line(line):
    line = " ".join(line.split())
    line = line.strip()
    line = line.replace("[_-]", ' ')
    line = line.replace("\'", '')
    line = line.replace("at&amp;t", "at&t")
    return line

#customized code
def write_corpus_to_file(fileName):
    F = open(fileName, "w")
    print("reading datasets ...")
    corpus_test = load_word_data(TEST_DATASET)
    corpus_train = load_word_data(TRAIN_DATASET)
    for l in corpus_test:
        F.write(clean_line(l))
    print("corpus_test written to '"+fileName+"'")
    for l in corpus_train:
        F.write(clean_line(l))
    print("corpus_train written to '"+fileName+"'")
    F.close()

write_corpus_to_file("./corpus.txt")
