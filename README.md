# Sentiment analysis with Twitter

Sentiment analysis with Twitter using [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset.

### Prerequisites

Things you need to install.

```
sudo apt-get install python3-pip
pip3 install tensorflow
pip3 install keras
pip3 install sklearn
pip3 install pandas
pip3 install tweepy
pip3 install nltk
nltk.download()

```

## Running

### Create model

Model and data options used can be modified from within lstm.py.

```
python3 lstm.py
```

### Obtain credentials from Twitter

Edit credentials.json with the tokens obtained in [Twitter Apps](https://apps.twitter.com/).

```
{
    "Consumer Key": "CONSUMER KEY",
    "Consumer Secret": "CONSUMER SECRET",
    "Access Token": "ACCESS TOKEN",
    "Access Token Secret": "ACCESS TOKEN SECRET"
}
```

### Get sentiment score

Sentiment score can be obtained by following the next structure 'sentiment_analysis.py "Query" number_tweets'. Example:

python3 sentiment_analysis.py "Machine Learning" 1000

## Authors

* **Jaime Céspedes Sisniega** - *Initial work* - [GitHub](https://github.com/jaimecespedes)
