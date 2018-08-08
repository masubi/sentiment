
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence, text
from keras.models import Model, Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Bidirectional, Input
from keras import regularizers
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.constraints import unitnorm, nonneg
from keras.optimizers import *
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import string
import pandas as pd
import datetime
import json
import re
from collections import defaultdict
from keras.callbacks import EarlyStopping, ModelCheckpoint, RemoteMonitor


# labeling variables
model_name = datetime.datetime.now().strftime("twitter_sentiment.%Y%m%dT%H%M.v")
np.random.seed(1337)  # for reproducibility

# data variables
TRAINING_EXAMPLES=1575116
SUBSET_SIZE = 1000000
HOLDOUT_SIZE = 200
max_features = 5003
maxlen = 100  # cut texts after this number of words (among top max_features most common words)

# TODO:  fill in w/ data
TRAIN_DATASET = [""] #e.g. ["/home/masubi/jupyter/training.1600000.processed.noemoticon.csv"]
TEST_DATASET = [""]  #e.g. ["/home/masubi/jupyter/testdata.manual.csv"]
DICTIONARY = ""      #e.g. "/home/masubi/jupyter/dictionary.json"
WORD_EMBEDDINGS = "" #e.g. "/home/masubi/fastText/result/twitter_corpus.vec"


#If use w/ existing models
load_weights=False
weights_file=""

# sample for training index
# I do this because the big twitter training set is sorted by label.
# If you take the first 100k examples they will all be one class.
# After you have your X & Y, the fit method will peform a shuffle for each epoch.
def file_gen(file_path):
    exclude = re.escape(re.sub(r"[\-\_]", "", string.punctuation))
    
    for i in file_path:
        df = pd.read_csv(i, header=None,
                         names=['label', 'id', 'date', 'query', "user", 'tweet'], encoding='latin-1')
        df = df.ix[(df.label.notnull()) & (df.tweet.str.count(" ") > 3), :]
        df['tweet'] = df.tweet.astype(str).str.lower()             .str.replace("[_-]", ' ')             .str.replace("\'", '')             .str.replace("\.", ' ')             .str.replace("at&amp;t", "at&t")
        yield df


# In[18]:


#
# word embeddings
#
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(WORD_EMBEDDINGS)

print("word_to_index shape: "+str(len(word_to_index)))
print("index_to_word shape: "+str(len(index_to_word)))
print("word_to_vec_map shape: "+str(len(word_to_vec_map)))


# In[4]:


#Write corpus to file

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
        
#write_corpus_to_file("./corpus.txt")

    


# In[5]:


# file_path = file to load
# return tweets tweets as indices
def get_data(file_path):
    df = pd.concat(file_gen(file_path))
    
    #  see list comprehension for how this works
    #    equivalent to below:
    #
    #  for line in df['tweet'].values
    #      for word in lines.split(" ")
    #          if(word in word_to_index.keys())
    #             word_to_index[word]
    records = [[word_to_index[word] for word in clean_line(line).split(" ") if (word in word_to_index.keys())]
               for line in df['tweet'].values]
    
    return sequence.pad_sequences(records, maxlen=maxlen), np.where(df.label.astype(int) >= 2, 1, 0)


    

print("Loading data...")
x_train_all, y_train_all = get_data(TRAIN_DATASET)
x_test_all, y_test_all = get_data(TEST_DATASET)
print("  orig x_train_all.shape: "+str(x_train_all.shape))
print("  orig y_train_all.shape: "+str(y_train_all.shape))
print("  orig x_test_all.shape: "+str(x_test_all.shape))
print("  orig y_test_all.shape: "+str(y_test_all.shape))
print("Loading data complete!")


# In[6]:


# ------------------------------------------
# Combine & Shuffling Train and Test Data
# ------------------------------------------
# Manual investigation shows that the test dataset contains negative, neutral, positive sentiments
# so need to combine and shuffle this data

# combine x and y cols
def concat_cols(x,y):
    y = y.reshape(y.shape[0],1)
    return np.concatenate((x,y), axis=1)

# split x and y
def split_cols(a):
    x_res = a[:,0:a.shape[1]-1]
    y_res = a[:,a.shape[1]-1:a.shape[1]]
    return x_res, y_res

# shuffle rows of a
def shuffle(a, subset_size):
    subset = a[np.random.choice(a.shape[0], subset_size, replace=False), :]
    return subset

# generates random subset of x,y
# x: x records
# y: y labels
# subset_size:  subset size
def generate_subset(x, y, subset_size):    
    combined = concat_cols(x,y)
    assert( subset_size < combined.shape[0])
    subset = shuffle(combined, subset_size)
    x_res = subset[:,0:x.shape[1]]
    y_res = subset[:,subset.shape[1]-1:subset.shape[1]]
    return x_res, y_res

print("Combine and Shuffle test w/ train Data")
test_subset_size=200 #size of test data to shuffle into 

#combine x_train_all and y_train_all
x_y_train = concat_cols(x_train_all, y_train_all)
print("  x_y_train.shape: "+str(x_y_train.shape))

#combine x_test_all, y_test_all
x_y_test = concat_cols(x_test_all, y_test_all)
print("  x_y_test.shape: "+str(x_y_test.shape))

# get subset of test
test_subset_to_shuffle_w_train = x_y_test[0:test_subset_size, :]
# get remaining x,y test
x_test, y_test = split_cols(x_y_test[test_subset_size:x_y_test.shape[0], :])

# combine and shuffle w/ train
shuffled_train = np.concatenate((x_y_train,test_subset_to_shuffle_w_train), axis=0)
shuffled_train = shuffle(shuffled_train, shuffled_train.shape[0])
print("  shuffled_train.shape:"+str(shuffled_train.shape))
x_train_all, y_train_all = split_cols(shuffled_train)


x_train_subset, y_train_subset = generate_subset(x_train_all, y_train_all, SUBSET_SIZE)
train_end_index = x_train_subset.shape[0]-HOLDOUT_SIZE
holdout_end_index = x_train_subset.shape[0]

print("Generate subset size: "+str(x_train_subset.shape))
x_train = x_train_subset[0:train_end_index,:]
y_train = y_train_subset[0:train_end_index,:]
print("  x_train.shape: "+str(x_train.shape))
print("  y_train.shape: "+str(y_train.shape))
#print("    x_train: "+str(x_train))
#print("    y_train: "+str(y_train))
x_holdout = x_train_subset[train_end_index:holdout_end_index,:]
y_holdout = y_train_subset[train_end_index:holdout_end_index,:]
print("  x_holdout.shape: "+str(x_holdout.shape))
print("  y_holdout.shape: "+str(y_holdout.shape))
#print("    x_holdout: "+str(x_holdout))
#print("    y_holdout: "+str(y_holdout))
print("positive examples in y_train:", y_train.sum())
print("positive examples in y_holdout:", y_holdout.sum())
print(len(x_train), 'train sequences')
print("Pad sequences (samples x time)")
print("One x_Training Example:\n", x_train[0, :])
print("One y_Training Example:\n", y_train[0, :])

print("x_test.shape: "+str(x_test.shape))
print("y_test.shape: "+str(y_test.shape))
print("One x_Test Example:\n", x_test[0, :])
print("One y_Test Example:\n", y_test[0])


# In[7]:


#for i in range(len(x_test)):
#    sentence = indicesToSentence(x_test[i])
#    print("i: "+str(i)+ ", "+sentence)


# In[8]:


#
# Experiment Metrics
#
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))


# In[9]:


print('Build model ... ' + model_name)

batch_size=32
epochs=20
optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        if(emb_matrix[index, :].shape == word_to_vec_map[word].shape):
            emb_matrix[index, :] = word_to_vec_map[word]
        else:
            print("shape mismatch")
            print("emb_matrix: "+str(emb_matrix[index, :].shape))
            print("word_to_vec_map[word]: "+str(word_to_vec_map[word].shape))
            print("word: "+word)

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


maxLen = maxlen#len(max(x_train, key=len).split())

# Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
#sentence_indices = np.zeros(input_shape, dtype='int32')
sentence_indices = Input(shape=(maxLen,))


# Create the embedding layer pretrained with GloVe Vectors (≈1 line)
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

# Propagate sentence_indices through your embedding layer, you get back the embeddings
embeddings = embedding_layer(inputs=sentence_indices)   

X = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(embeddings)
X = Bidirectional(LSTM(128, kernel_regularizer=regularizers.l2(0.01)))(X)
X = Dense(128, activation='relu')(X)
X = Dense(1, activation='sigmoid')(X)

# Create Model instance which converts sentence_indices into X.
model = Model(inputs=[sentence_indices], outputs=[X])

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', f1, precision, recall])
print(model.summary())

if(load_weights):
    print("loading weights from: "+weights_file)
    model.load_weights(weights_file)
    print("weights: ")
    print(str(model.get_weights()))
    print("loading weights complete!")

print('Build model complete')


# In[10]:


print('Training...')

checkpointer = ModelCheckpoint(filepath='./'+model_name+'-weights.hdf5', verbose=2, save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs/'+model_name, 
                          histogram_freq=0,
                          write_graph=True, 
                          write_images=True)


history = model.fit(x_train, 
                    y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=1, 
                    validation_data=(x_holdout, y_holdout), 
                    callbacks=[checkpointer, tensorboard])
    
print('Training complete')


# In[11]:


print('Testing...')

print('  model.metrics_names: '+str(model.metrics_names))
print('x_test.shape:', x_test.shape)
loss, accuracy, f1, precision, recall = model.evaluate(x_test, y_test, batch_size=32)
metrics = {
    'loss':loss,
    'accuracy':accuracy,
    'f1': f1,
    'precision': precision,
    'recall': recall
}


print('loss: '+str(loss))
print('accuracy: '+str(accuracy))
print('f1: '+str(f1))
print('precision: '+str(precision))
print('recall: ' + str(recall))


# In[12]:


#
# debug predictions
#
def indicesToSentence(indices):
    result = ""
    for i in range(len(indices)):
        if(indices[i] != "0" and indices[i] in index_to_word.keys()): 
            result=result+" "+str(index_to_word[indices[i]])
    return result

error_file = model_name+"_test_errors.txt"
def print_errors(x_test, y_test, model):   
    F = open(error_file, "w")
    pred = model.predict(x_test)
    errorCount = 0
    for i in range(len(x_test)):
        sentence = indicesToSentence(x_test[i])
        prediction = pred[i]
        actual = y_test[i]
        if(abs(prediction - actual) > .5):
            resultStr = "i: "+str(i)+" pred: "+str(prediction)+" actual: "+str(actual) + " sentence: "+sentence
            print(resultStr)
            F.write(resultStr+"\n")
            errorCount=errorCount+1
    print("-------------------------------")
    print("errorCount: "+str(errorCount))
    print("total: "+str(len(x_test)))
    print("sample test sentence: "+str(indicesToSentence(x_test[0])))   
    F.close()

success_file = model_name+"_test_success.txt"
def print_successes(x_test, y_test, model):   
    F = open(success_file, "w")
    pred = model.predict(x_test)
    errorCount = 0
    for i in range(len(x_test)):
        sentence = indicesToSentence(x_test[i])
        prediction = pred[i]
        actual = y_test[i]
        if(abs(prediction - actual) < .5):
            resultStr = "i: "+str(i)+" pred: "+str(prediction)+" actual: "+str(actual) + " sentence: "+sentence
            print(resultStr)
            F.write(resultStr+"\n")
            errorCount=errorCount+1
    print("-------------------------------")
    print("sucessCount: "+str(errorCount))
    print("total: "+str(len(x_test)))
    print("sample test sentence: "+str(indicesToSentence(x_test[0])))  
    F.close()
    
print_errors(x_test, y_test, model)    
print_successes(x_test, y_test, model)


# In[13]:


print('Save Model')
model.save_weights(model_name+".hdf5", overwrite=True)
yaml_string = model.to_yaml()
open(model_name+'.yaml', 'w+').write(yaml_string)

