# speech emotion from acoustic and text
# text weights = FastText
# coded by Bagus Tris Atmaja, bagus@ep.its.ac.id

import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras.backend as K

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU, BatchNormalization
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping
import codecs

import io
import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss

# load feature and labels
feat = np.load('/home/s1820002/atsit/data/feat_34_hfs.npy')
vad = np.load('/home/s1820002/IEMOCAP-Emotion-Detection/y_egemaps.npy')
vad = vad[:,0].reshape(-1,1)

# remove outlier, < 1, > 5
vad = np.where(vad==5.5, 5.0, vad)
vad = np.where(vad==0.5, 1.0, vad)

scaled_vad = True
scaled_feature = False

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaled_feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

# text extraction
path = '/home/s1820002/IEMOCAP-Emotion-Detection/'
data = np.load(path+'data_collected_10039.pickle', allow_pickle=True)
text = [t['transcription'] for t in data]

# load fastext model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
MAX_SEQUENCE_LENGTH = len(max(token_tr_X, key=len))
x_train_text = []
x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
nb_words = len(word_index) +1
print('Found %s unique tokens' % len(word_index))

# load model
EMBEDDING_DIM = 300
file_loc = '/home/s1820002/fasttext/crawl-300d-2M-subword.vec'
print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding
#
f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))


# API model
def model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = CuDNNLSTM(feat.shape[2], return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=False)(net_speech)
    model_speech = Dense(64)(net_speech)
    #model_speech = Dropout(0.1)(net_speech)
    
    #text network
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net_text = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(input_text)
    net_text = CuDNNLSTM(300, return_sequences=True)(net_text)
    net_text = CuDNNLSTM(256, return_sequences=True)(net_text)
    net_text = CuDNNLSTM(256, return_sequences=False)(net_text)
    net_text = Dense(64)(net_text)
    model_text = Dropout(0.4)(net_text)

    # combined model
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(64, activation='relu')(model_combined)
    model_combined = Dense(32, activation='relu')(model_combined)
    model_combined = Dropout(0.4)(model_combined)
#    target_names = ('v', 'a', 'd')
#    model_combined = [Dense(1, name=name)(model_combined) for name in target_names]
    model_combined = Dense(1)(model_combined)
    
    model = Model([input_speech, input_text], model_combined) 
    model.compile(loss=ccc_loss,
                  #loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='rmsprop', metrics=[ccc])
    return model


model = model(0.7, 0.2, 0.1)
model.summary()

# 7869 first data of session 5 (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit([feat[:7869], x_train_text[:7869]], 
                  vad[:7869], batch_size=256, #best:8
                  validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                  callbacks=[earlystop])
metrik = model.evaluate([feat[7869:], x_train_text[7869:]], vad[7869:])
print("CCC: ", metrik) # np.mean(metrik[-3:]))
