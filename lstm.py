from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from preprocess_tweet import preprocess_tweet
import pandas as pd
import pickle

corpus = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=['Sentiment', 'Id', 'Date',
                                                                                         'Query', 'Username', 'Text'])

corpus = corpus.drop(['Id', 'Date', 'Query', 'Username'], axis=1)
df = shuffle(corpus)
df = df[:200000]

df['Text'] = df['Text'].apply(preprocess_tweet)

num_words = 5000
tk = Tokenizer(num_words=num_words, split=' ')
tk.fit_on_texts(df['Text'].values)
X = tk.texts_to_sequences(df['Text'].values)
X = pad_sequences(X)

model = Sequential()
model.add(Embedding(num_words, 256, input_length=X.shape[1]))
model.add(LSTM(512, recurrent_dropout=0.4, dropout=0.4, return_sequences=True))
model.add(LSTM(64, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

Y = pd.get_dummies(df['Sentiment'])[0].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model.fit(X_train, y_train, epochs=3, batch_size=256, validation_data=(X_test, y_test))

with open('tk.pickle', 'wb') as handle:
    pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Tokenizer saved')

with open('sentence_length.pickle', 'wb') as handle:
    pickle.dump(X.shape[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Sentence length saved')

model.save('sentiment_model')
print('Model saved')
