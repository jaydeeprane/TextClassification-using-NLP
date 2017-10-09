from csv import DictReader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
import nltk 
from collections import Counter
from random import *

def read_data(name):
    text, targets = [], []

    with open('../data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            text.append(item['text'].decode('utf8'))
            targets.append(item['category'])
    return text, targets


def make_prediction(x_train, x_valid, y_train, y_valid):

    model = make_pipeline(
            TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=600, sublinear_tf=True),
           LogisticRegression(),
        ).fit(x_train, y_train)

    prediction = list(model.predict(x_valid))

    stopwords = set(nltk.corpus.stopwords.words('english'))
    for i, line in enumerate(x_valid):
        words = [w.lower() for w in line.strip().split() if (w not in stopwords and len(w)>=3)]
        if('millitary' in words or 'veteran' in words or 'marine' in words or 'army' in words or 'military' in words):
            prediction[i] = 'military'
        if('baseball' in words or 'gym' in words or 'soccer' in words or 'football' in words):
            prediction[i] = 'sports'
        if('jesus' in words or 'bible' in words or 'god' in words or 'christian' in words):
            prediction[i] = 'faith'
        if('heels' in words or 'flats' in words or 'hair' in words or 'jeans' in words or 'dress' in words):
            prediction[i] = 'fashion'
        if('tattoo' in words or 'tattoos' in words):
            prediction[i] = 'tatoos'
    print 'macro f1:', f1_score(y_valid, prediction, average ='macro')
    

def main():

    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')
    
    for i in xrange(5):
        seed = randint(0, 123)
        x_train, x_valid, y_train, y_valid = train_test_split(text_train, targets_train, test_size=0.3, random_state=seed)
        make_prediction(x_train, x_valid, y_train, y_valid)
    make_prediction(text_train,text_test,targets_train,targets_test)

if __name__ == "__main__":
    main()