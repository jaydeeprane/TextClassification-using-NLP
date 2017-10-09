from csv import DictReader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import numpy as np


def read_data(name):
    text, targets = [], []

    with open('../data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            # Only including the top 4 categories
            if((item['category'] == 'personal' or item['category'] == 'meetup' or item['category'] == 'misc' 
                or item['category'] == 'relationships')):
                text.append(item['text'].decode('utf8'))
                targets.append(item['category'])

    return text, targets


def main():
    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')


    model = make_pipeline(
        TfidfVectorizer(max_features=600, sublinear_tf=True, stop_words='english', strip_accents='ascii'),
        LogisticRegression(),
    ).fit(text_train, targets_train)

    prediction = list(model.predict(text_test))

    print 'macro f1:', f1_score(targets_test, prediction, average='macro')
    # print 'weighted f1:', f1_score(targets_test, prediction, average='weighted')


if __name__ == "__main__":
    main()
