from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocess_tweet import preprocess_tweet
from twitter import TwitterSearch
import pandas as pd
import pickle
import sys

if __name__ == "__main__":
	query = str(sys.argv[1])
	num_tweets = int(sys.argv[2])
	model = load_model('sentiment_model')

	# loading tokenizer
	with open('tk.pickle', 'rb') as handle:
		tk = pickle.load(handle)

	# loading sentence length
	with open('sentence_length.pickle', 'rb') as handle:
		sentence_length = pickle.load(handle)

	df = TwitterSearch().search(query, num_tweets=num_tweets)

	df = df[[True if query in text else False for text in df.text]]

	df['text_processed'] = df['text'].apply(preprocess_tweet)
	
	df.drop_duplicates(subset=['text_processed'], keep=False, inplace=True)
	df.dropna(axis=0, how='any', inplace=True)
	df.reset_index(drop=True, inplace=True)

	pos_count, neg_count, neu_count = 0, 0, 0

	#results = pd.DataFrame(columns=['Tweet', 'Pos score', 'Neg score'])

	threshold = 0.65

	for i, tweet in df.iterrows():
		x_new = pad_sequences(tk.texts_to_sequences([tweet['text_processed']]), sentence_length)
		pred = model.predict(x_new)[0][0]

		neg = pred
		pos = 1-pred

		print('{} - {}'.format(i+1, tweet['text']))
		print('Neg score: {:.4}  Pos score: {:.4}\n'.format(neg, pos))

		#results = results.append({'Tweet': tweet, 'Pos score': pos, 'Neg score': neg}, ignore_index=True)

		if neg >= threshold:
			neg_count += 1
		elif pos >= threshold:
			pos_count += 1
		else:
			neu_count += 1

	total = neg_count + pos_count + neu_count

	print('Negative: {} - {:.4}%'.format(neg_count, (neg_count/total)*100))
	print('Positive: {} - {:.4}%'.format(pos_count, (pos_count/total)*100))
	print('Neutral: {} - {:.4}%'.format(neu_count, (neu_count/total)*100))
