import json
import pandas as pd
import tweepy


class TwitterSearch:

    def __init__(self):
        credentials = json.load(open('credentials.json'))
        self.auth = tweepy.OAuthHandler(credentials['Consumer Key'], credentials['Consumer Secret'])
        self.auth.set_access_token(credentials['Access Token'], credentials['Access Token Secret'])
        self.api = tweepy.API(self.auth)

    def search(self, query, num_tweets=1000):
        tweets = {'date': [], 'text': []}
        for tweet in tweepy.Cursor(self.api.search, q=query, lang='en', result_type='recent',
                                   tweet_mode='extended').items(num_tweets):
            tweets['date'].append(tweet.created_at)
            tweets['text'].append(tweet.full_text)

        df = pd.DataFrame(tweets)
        df.sort_values(by='date', inplace=True, ascending=True)
        return df
