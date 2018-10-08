#import everything thats needed
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
import pickle as pkl
import os
import sys


WEIGHTS_SAVE_DIR='weights_checkpoints/'
CHECKPOINT_NAME= WEIGHTS_SAVE_DIR+"/with-20k-features-{epoch:02d}.hdf5"
TRAIN_TEST_SPLIT='data/final_data.pkl'

if not os.path.exists(WEIGHTS_SAVE_DIR):
	os.mkdir(WEIGHTS_SAVE_DIR)

#create the training class to keep everything organized
class Train:
    def __init__(self):
        self.load_dataset()
        self.model=None
        self.checkpoint=ModelCheckpoint(CHECKPOINT_NAME, monitor='val_acc', verbose=1, save_best_only=False)
        

    def load_best_weights(self):
        #get every weight file in the WEIGHTS_SAVE_DIR
        files_list = [f for f in os.listdir(WEIGHTS_SAVE_DIR) if f.endswith('hdf5')]
        
        #check if its the first run and nothing is daved yet
        if len(files_list)==0:
            print('no weight files to load, model weights not updated')
            return
        
        #gets best weights based on name of the file. we get last 2 characters
        fname = max(ls, key=lambda s: s.split('-')[-1][:2])
        
        #load the model with weights
        print('using weights from file: ', fname)
        self.model.load_weights(WEIGHTS_SAVE_DIR+ fname)
        

    
    def create_model(self, input_shape, max_features=20000):
        embed_dim = 128
        lstm_out = 196
        
        #create a checkpoint to moniter the accuracy and save only if improvement is observed
        
        #create the very simple model
        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=input_shape, dropout=0.2))
        model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        
        return model

    def load_dataset(self):
        #loads the preprocessed data
        if not os.path.isfile(TRAIN_TEST_SPLIT): 
            print('preprocessed dataset not found\nrun data_preprocess.py first...')
            quit()
        
        with open(TRAIN_TEST_SPLIT, 'rb') as handle:
            self.X_train, self.X_test, self.Y_train, self.Y_test  = pkl.load(handle)


    
    def train(self, epochs):        
        #use already cleaned and split data else clean and split and save it for later
        if not os.path.exists(TRAIN_TEST_SPLIT):
            print('preprocessed dataset not found\nrun data_preprocess.py first...')
            quit()
        
        #model creation
        self.model = self.create_model(input_shape=self.X_train.shape[1], max_features=20000)

        callbacks_list = [self.checkpoint]
        
        print(f'training started with {epochs}\n')
        self.history = self.model.fit(self.X_train, self.Y_train, validation_split=0.1, 
                                      epochs=epochs, batch_size=64,
                                      callbacks=callbacks_list, verbose=1)

if __name__ == "__main__":
    train = Train()
    train.train(epochs=int(sys.argv[1]))