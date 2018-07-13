import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import pickle as pkl
from tkinter import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint


class Train:
    def __init__(self):
        self.df = self.load_dataset()

        pass

    def load_best_weights(self, model):
        ls = [l for l in os.listdir('weights_checkpoints/') if l.endswith('hdf5')]
        f = max(ls, key=lambda s: s.split('-')[-1][:2])
        model.load_weights('weights_checkpoints/' + f)
        print('weights loaded:', f)
        return model

    def create_model(self, input_shape, max_features=20000):
        embed_dim = 128
        lstm_out = 196
        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=input_shape, dropout=0.2))
        model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        filepath = "weights_checkpoints/weights-improvement-20k-features-{epoch:02d}.hdf5"
        # elif rev == 2:
        #     model.add(Embedding(max_features, embed_dim, input_length=input_shape, dropout=0.2))
        #     model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
        #     model.add(Dense(5, activation='softmax'))
        #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #     print(model.summary())
        #     filepath = "weights_checkpoints/rev2/weights-improvement-{epoch:02d}.hdf5"
        #     print('does not exist , yet')
        return model, ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)

    def load_dataset(self):
        with open('dataset/twitter_like.pkl', 'rb') as handle:
            dataset = pkl.load(handle)
        return dataset

    def load_tokenizer(self):
        with open('misc/255_tokenizer.pkl', 'rb') as handle:
            self.tokenizer = pkl.load(handle)

    def train(self, epochs):
        self.load_tokenizer()
        if os.path.exists('dataset/test_train.pkl'):
            X_train, X_test, Y_train, Y_test = self.load_split_data('dataset/test_train.pkl')
        else:
            X = self.tokenizer.texts_to_sequences(self.df.reviewText.values)
            X = pad_sequences(X)
            Y = pd.get_dummies(self.df.overall)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
            self.save_split_data('dataset/test_train.pkl', (X_train, X_test, Y_train, Y_test))
        self.model, checkpoint = self.create_model(input_shape=X_train.shape[1], max_features=20000)

        callbacks_list = [checkpoint]
        self.history = self.model.fit(X_train, Y_train, validation_split=0.33, epochs=epochs, batch_size=64,
                                      callbacks=callbacks_list, verbose=1)

    def load_split_data(self, file):
        with open(file, 'rb') as handle:
            return pkl.load(handle)

    def save_split_data(self, file, data):
        with open(file, 'wb') as handle:
            return pkl.dump(data, handle)


if __name__ == "__main__":
    train = Train()
    train.train(epochs=1)
