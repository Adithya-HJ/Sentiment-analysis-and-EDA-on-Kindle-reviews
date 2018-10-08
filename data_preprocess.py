import pandas as pd 
import gzip 
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import string
import pickle as pkl
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os

DATA_FOLDER='data/'
MISC_FOLDER='misc/'
trans_vals='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
os.mkdir(DATA_FOLDER)
os.mkdir(MISC_FOLDER)

#borrowed from the same page
def parse(path):
    g = gzip.open(path, 'rb') 
    for l in g:
        yield eval(l) 

def getDF(path):
    i = 0 
    df = {} 
    for d in tqdm(parse(path)):
        df[i] = d 
        i += 1 
    return pd.DataFrame.from_dict(df, orient='index') 


def clean_data(dataset):
    with_expansions=list()
    trans_table=str.maketrans('','',trans_vals)
    dataset.reviewText = dataset.reviewText.apply(lambda x: x.lower())
    for line in tqdm(dataset.reviewText.values):
        clean_line=[w.translate(trans_table) for w in line.split(' ')]
        var=re.sub(' +',' ',' '.join(clean_line))
        with_expansions.append(var)
    return with_expansions

# loads data from a pickle file
def load_dataset(filename):
    with open(DATA_FOLDER+filename,'rb') as handle:
        dataset = pkl.load(handle)
    return dataset


# saves updated dataset
def save_dataset(dataset,filename):
    with open(DATA_FOLDER+filename, 'wb') as handle:
        pkl.dump(dataset, handle)

# save tokenizer
def save_tokenizer(tokenizer):
    with open('misc/tokenizer.pkl', 'wb') as handle:
        pkl.dump(tokenizer, handle)



df = getDF('raw_data/reviews_Kindle_Store_5.json.gz') 

data=df[['reviewText','overall']].copy()
del df

data.overall=data.overall.astype('int')

mask=(data.reviewText.str.len()>50) & (data.reviewText.str.len()<300)
data=data.loc[mask]

data['clean_text']=clean_data(data)
data=data.drop(['reviewText'],axis=1)

save_dataset(data,'std-clean-data.pkl')

tokenizer = Tokenizer(num_words=20000, split=' ')
tokenizer.fit_on_texts(data.clean_text.values)
save_tokenizer(tokenizer)

data['tokenized'] = tokenizer.texts_to_sequences(data.clean_text.values)

del tokenizer

X = pad_sequences(data.tokenized,maxlen=80,padding='post')
y= pd.get_dummies(data.overall)
X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=0.3,random_state=1337)
save_dataset([X_train, X_test, Y_train, Y_test],'final_data.pkl')

print('all done and set')
