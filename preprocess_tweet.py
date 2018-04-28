import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_tweet(text):
    text = re.sub(r'@[A-Za-z0-9]+|http[s]?://[A-Za-z0-9./]+|RT @[A-Za-z0-9 ]+:', '', text)

    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

    words = [word for word in word_tokenize(text) if word not in stop_words]

    return ' '.join(words)
