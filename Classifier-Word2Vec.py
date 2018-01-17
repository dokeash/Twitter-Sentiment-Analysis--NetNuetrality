import json
import pandas as pd
from nltk import tokenize
import pandas as pd


tweet_files = [] #list of files
tweets_data = []
for file in tweet_files:
    with open(file, 'r') as f:
        for line in f.readlines():
            tweets_data.append(json.loads(line))


#--------------------------------------------preprocessing methods-----------------------------------
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

#Json to dataframe
def populate_tweet_df(tweets):
    df = pd.DataFrame()
    df['text'] = list(map(lambda tweet: tweet['text'], tweets))
    return df

#---------------------------------------------------------------------------------------------


import numpy as np  # Make sure that numpy is imported
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average

    featureVec = np.divide(featureVec,nwords)

    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above.
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

print("Total Tweets $:",len(tweets_data))
data=populate_tweet_df(tweets_data)

#-----------------------------------------Setting up Ground Truth--------------------------
#Sentment Analysing by using existing library
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

data['sentiment_compound_polarity']=data.text.apply(lambda x:sid.polarity_scores(x)['compound'])
data['sentiment_neutral']=data.text.apply(lambda x:sid.polarity_scores(x)['neu'])
data['sentiment_negative']=data.text.apply(lambda x:sid.polarity_scores(x)['neg'])
data['sentiment_pos']=data.text.apply(lambda x:sid.polarity_scores(x)['pos'])
data['sentiment_type']=''
data.loc[data.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
data.loc[data.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
data.loc[data.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'

#Train-Test Split
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
train,test = train_test_split(data,test_size=0.3,random_state=42)


from gensim.models import KeyedVectors
#KeyedVectors.load_word2vec_format
model = KeyedVectors.load("300features_40minwords_10_2_context")

clean_train_reviews = []
for review in train["text"]:
    clean_train_reviews.append( review_to_wordlist(review, remove_stopwords=True ))

print("len:",len(clean_train_reviews))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, 300)

print ("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["text"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, 300)

list_of_labels=['POSITIVE','NEUTRAL','NEGATIVE']

#-----------------------------------------building model-----------------------
# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
forest = RandomForestClassifier( n_estimators = 100 )

print ("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, list(train["sentiment_type"].values))

# Test & extract results
predictions = forest.predict(testDataVecs )
accuracy = accuracy_score(predictions,list(test['sentiment_type'].values))

print("Accuracy: ",accuracy)
precision = precision_score(list(test['sentiment_type'].values), predictions, average=None, pos_label=None, labels=list_of_labels)
recall = recall_score(list(test['sentiment_type'].values), predictions, average=None, pos_label=None, labels=list_of_labels)
accuracy = accuracy_score(list(test['sentiment_type'].values), predictions)
f1 = f1_score(list(test['sentiment_type'].values), predictions, average=None, pos_label=None, labels=list_of_labels)
print("=================== Results ===================")
print("            Negative     Neutral     Positive")
print("F1       " + str(f1))
print("Precision" + str(precision))
print("Recall   " + str(recall))
print("Accuracy " + str(accuracy))
print("===============================================")

#-----------------------------------------Confusion Matrix---------------------------
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(list(test['sentiment_type'].values), predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=list_of_labels,
                      title='Confusion matrix, without normalization')

plt.show()