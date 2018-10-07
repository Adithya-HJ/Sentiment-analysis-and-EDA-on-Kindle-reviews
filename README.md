This is a project that aims to classify text i.e we do sentiment analysis here. Sentiment analysis gives us a very short summary that is the emotion of the text.
The dataset we used in this project can be found here

http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz

There are many other datasets you can try on the same site 

http://jmcauley.ucsd.edu/data/amazon

We used LSTM's in our model you can change the model to your liking.

To run run this project
first download the dataset and place it in `raw_data` folder

First run `data_preprocess.py`

        python data_preprocess.py

This basically does some preprocessing and saves the tokenizer and the dataset into pickle files  to make the training time faster.

to train the network use `rnn_trainer.py`


        python rnn_trainer.py epochs
            replace epochs with how many epochs you want to train.

There is also corresponding Jupyter Notebook files.

some cleaning left to do.
work in progress.....
