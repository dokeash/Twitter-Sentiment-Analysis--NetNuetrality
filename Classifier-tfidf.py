import json
import pandas as pd
from nltk import tokenize
import pandas as pd
import matplotlib.pyplot as plt

tweet_files = [] #list of files
tweets_data = []
for file in tweet_files:
    with open(file, 'r') as f:
        for line in f.readlines():
            tweets_data.append(json.loads(line))


# Json to dataframe
def populate_tweet_df(tweets):
    df = pd.DataFrame()
    df['text'] = list(map(lambda tweet: tweet['text'], tweets))
    return df


# ------------------------------------------preprocessing-------------------------#

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re

networds = ['internet','netnuetrality']

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def preprocessor(text):
    words = text.split()
    stops = set(stopwords.words("english"))
    del words[0]
    words = [re.sub('[^0-9a-zA-Z]+', '', w.replace('#', '').lower()) for w in words if
             not w.startswith('http') and not w.startswith('@')]
    finalwords = [w for w in words if (not w.isdigit() and w not in networds and w not in stops)]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(finalwords, stemmer)
    return ' '.join(stemmed)


data = populate_tweet_df(tweets_data)
print("Total Tweets $:", len(tweets_data))
data['text'] = data['text'].apply(preprocessor)


# --------------------------categorizing tweets-Setting up Ground Truth---------------------------------------------

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

data['sentiment_compound_polarity'] = data.text.apply(lambda x: sid.polarity_scores(x)['compound'])
data['sentiment_neutral'] = data.text.apply(lambda x: sid.polarity_scores(x)['neu'])
data['sentiment_negative'] = data.text.apply(lambda x: sid.polarity_scores(x)['neg'])
data['sentiment_pos'] = data.text.apply(lambda x: sid.polarity_scores(x)['pos'])
data['sentiment_type'] = ''
data.loc[data.sentiment_compound_polarity > 0, 'sentiment_type'] = 'POSITIVE'
data.loc[data.sentiment_compound_polarity == 0, 'sentiment_type'] = 'NEUTRAL'
data.loc[data.sentiment_compound_polarity < 0, 'sentiment_type'] = 'NEGATIVE'
print(data.head(2))

#Check for imbalanced data with sentiment distribution graph. If not, start training
# -------------training------------------------------------------------------------------------
from sklearn.cross_validation import train_test_split
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time
from collections import Counter

train, test = train_test_split(data, test_size=0.3, random_state=42)


train_clean_tweet = []
for tweets in train['text']:
    train_clean_tweet.append(tweets)
test_clean_tweet = []
for tweets in test['text']:
    test_clean_tweet.append(tweets)

v = CountVectorizer(analyzer="word")
train_features = v.fit_transform(train_clean_tweet)
test_features = v.transform(test_clean_tweet)

Classifiers = [
    LogisticRegression(C=0.000000001, solver='liblinear', max_iter=200),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]

dense_features = train_features.toarray()
dense_test = test_features.toarray()
Accuracy = []
Model = []
list_of_labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features, train['sentiment_type'])
        predictions = fit.predict(test_features)


    except Exception:
        fit = classifier.fit(dense_features, train['sentiment_type'])
        predictions = fit.predict(dense_test)

    accuracy = accuracy_score(predictions, test['sentiment_type'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of ' + classifier.__class__.__name__ + ' is ' + str(accuracy))

    precision = precision_score(test['sentiment_type'], predictions, average=None, pos_label=None,
                                labels=list_of_labels)
    recall = recall_score(test['sentiment_type'], predictions, average=None, pos_label=None, labels=list_of_labels)
    f1 = f1_score(test['sentiment_type'], predictions, average=None, pos_label=None, labels=list_of_labels)
    print("=================== Results ===================")
    print("            Negative     Neutral     Positive")
    print("F1       " + str(f1))
    print("Precision" + str(precision))
    print("Recall   " + str(recall))
    print("Accuracy " + str(accuracy))
    print("===============================================")

Index = [1, 2, 3, 4, 5, 6]
plt.bar(Index, Accuracy)
plt.xticks(Index, Model, rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models')
plt.show()
