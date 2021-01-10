# dimensional speech emotion from acoustic and text
# text weight = BERT
# coded by Bagus Tris Atmaja, bagus@ep.its.ac.id
# changelog:
# 2020-02-12: initial code, modified from ter_bert_lstm.py

import os

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU, BatchNormalization
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping
#from keras_bert import extract_embeddings
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert import BertTokenizer

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load feature and labels
feat = np.load('/home/s1820002/atsit/data/feat_34_hfs.npy')


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

path = '/home/s1820002/IEMOCAP-Emotion-Detection/'
x_train_text = np.load(path+'x_train_text.npy')


def get_bert_embed_matrix():
    bert = BertModel.from_pretrained(BERT_FP)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat


def get_bert_embed_matrix():
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat


g_word_embedding_matrix = get_bert_embed_matrix()

vad = np.load(path+'y_egemaps.npy')

# remove outlier, < 1, > 5
vad = np.where(vad==5.5, 5.0, vad)
vad = np.where(vad==0.5, 1.0, vad)

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

# other parameters
MAX_SEQUENCE_LENGTH = 554
EMBEDDING_DIM = 768 #1024
nb_words = 3438

# Keras API model
def model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = CuDNNLSTM(feat.shape[2], return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=False)(net_speech)
    #model_speech = Flatten()(net_speech)
    model_speech = Dense(64)(net_speech)
    #model_speech = Dropout(0.1)(net_speech)
    
    #text network
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net_text = Embedding(*g_word_embedding_matrix.shape,
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
    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name)(model_combined) for name in target_names]

    model = Model([input_speech, input_text], model_combined) 
    model.compile(loss=ccc_loss,
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='rmsprop', metrics=[ccc])
    return model


model = model(0.7, 0.2, 0.1)
model.summary()

# 7869 first data of session 5 (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit([feat[:7869], x_train_text[:7869]], 
                  vad[:7869].T.tolist(), batch_size=256, #best:8
                  validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                  callbacks=[earlystop])
metrik = model.evaluate([feat[7869:], x_train_text[7869:]], vad[7869:].T.tolist())
print("CCC: ", metrik[-3:]) # np.mean(metrik[-3:]))
print("CCC_mean: ", np.mean(metrik[-3:]))
