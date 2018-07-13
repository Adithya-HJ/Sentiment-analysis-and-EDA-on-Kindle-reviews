import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
from tkinter import Frame
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import pickle as pkl
from keras.preprocessing.sequence import pad_sequences
import operator
import os
from tkinter import ttk
from keras.callbacks import ModelCheckpoint
import io
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import ImageTk,Image

f = io.StringIO()

def load_best_weights(model):
    ls = [l for l in os.listdir('weights_checkpoints/') if l.endswith('hdf5')]
    f = max(ls, key=lambda s: s.split('-')[-1][:2])
    model.load_weights('weights_checkpoints/' + 'weights-improvement-20k-features-03.hdf5')
    print('weights loaded:', f)
    return model


def create_model(input_shape=255, max_features=20000):
    # model
    embed_dim = 128
    lstm_out = 196
    filepath = "weights_checkpoints/weights-improvement-20k-features-{epoch:02d}.hdf5"

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=input_shape))
    model.add(SpatialDropout1D(rate=0.2))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model, [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)]


def load_tokenizer():
    with open('misc/255_tokenizer.pkl', 'rb') as handle:
        return pkl.load(handle)


class Application(Frame):
    def __init__(self, master=None):
        master.title('Sentiment Analyser UI')
        master.geometry('800x300')
        Frame.__init__(self, master)
        self.df = self.load_dataset()
        self.model, self.checkpoint = create_model()
        self.model = load_best_weights(self.model)
        self.tokenizer=self.load_tokenizer()
        self.parent = master
        self.pack()
        self.one=ImageTk.PhotoImage(Image.open('one.png'))
        self.two=ImageTk.PhotoImage(Image.open('two.png'))
        self.three=ImageTk.PhotoImage(Image.open('three.png'))
        self.four=ImageTk.PhotoImage(Image.open('four.png'))
        self.five=ImageTk.PhotoImage(Image.open('five.png'))
        self.createWidgets()

        pass

    def load_dataset(self):
        with open('dataset/twitter_like.pkl', 'rb') as handle:
            dataset = pkl.load(handle)
        return dataset

    def createWidgets(self):
        self.tabControl = ttk.Notebook(self.parent)
        self.predictionTab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.predictionTab, text='Predict')
        self.train_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.train_tab, text='Train')
        self.tabControl.pack(expand=1, fill='both')

        predictionText = tk.StringVar()
        predictionText.trace("w",
                             lambda name, index, mode, predictionText=predictionText: self.callback(predictionText))
        tk.Label(self.predictionTab, text='Input a string:').pack(side=tk.TOP)
        tk.Entry(self.predictionTab, textvariable=predictionText).pack(side=tk.TOP, fill='x', padx=20, pady=20)
        tk.Label(self.predictionTab,textvariable=predictionText,wraplength=700, text='Entered string is:\n').pack(expand=0,fill='y', padx=20, pady=20)
        self.std_out_dump=tk.Text(self.train_tab)
        self.std_out_dump.config(state=tk.DISABLED)
        self.predictionOutText = tk.StringVar()
        self.image=tk.Label(self.predictionTab,image=self.one)
        self.image.pack(side=tk.RIGHT)
        predition = tk.Label(self.predictionTab, textvariable=self.predictionOutText)
        predition.pack(side=tk.BOTTOM)
        tk.Label(self.train_tab,
                 text='Note:\n Best weights are loaded based on previous training. Training will take an hour or more based on where it is being run').pack(
            side=tk.TOP)
        self.start_button=tk.Button(self.train_tab,text='Start training',command=self.start_training).pack()

    def callback(self, predictionText):
        if predictionText.get()[-1] == ' ':
            print(predictionText.get())
            output,img = self.make_prediction((predictionText.get()))
            self.predictionOutText.set(output)
            self.image.configure(image=img)

    def make_prediction(self, text):
        self.tokenizer = load_tokenizer()
        _X = self.tokenizer.texts_to_sequences([text])
        _X = pad_sequences(_X, maxlen=255)
        op = list(self.model.predict(_X).ravel())
        i, v = max(enumerate(op), key=operator.itemgetter(1))
        i += 1
        res = '\n\n\npredicted rating is:\t' + str(i) + '\n\n'
        if i == 1:
            img=self.one
            sent = 'poor,very bad, Hated it'
        elif i == 2:
            img=self.two
            sent = 'not bad, i don\'t prefer it'
        elif i == 3:
            img=self.three
            sent = 'neutral, Can\'t say anything'
        elif i == 4:
            img=self.four
            sent = 'good, satisfactory'
        else:
            img=self.five
            sent = 'Excellent, Loved it'
        return str('predicted rating is' + str(i) + ' ' + sent),img

    def start_training(self):
        if os.path.exists('dataset/test_train.pkl'):
            X_train, X_test, Y_train, Y_test = self.load_split_data('dataset/test_train.pkl')
        else:
            X = self.tokenizer.texts_to_sequences(self.df.reviewText.values)
            X = pad_sequences(X,maxlen=255)
            Y = pd.get_dummies(self.df.overall)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
            self.save_split_data('dataset/test_train.pkl', (X_train, X_test, Y_train, Y_test))
        with redirect_stdout(f):
            X_train, X_test, Y_train, Y_test = self.load_split_data('dataset/test_train.pkl')
            self.model.fit(X_train,Y_train,batch_size=24,callbacks=self.checkpoint,epochs=1,verbose=1)
            self.std_out_dump.insert(tk.END,f.getvalue())
    def load_split_data(self, file):
        with open(file, 'rb') as handle:
            return pkl.load(handle)

    def save_split_data(self, file, data):
        with open(file, 'wb') as handle:
            return pkl.dump(data, handle)
    def load_tokenizer(self):
        with open('misc/255_tokenizer.pkl', 'rb') as handle:
            self.tokenizer = pkl.load(handle)




if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
