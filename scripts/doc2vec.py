from csv import DictReader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import nltk
import random
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import sklearn.linear_model
import sklearn.naive_bayes

def read_data(name):
    text, targets,text_sl = [], [], []

    with open('../data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
                targets.append(item['category'])
                text.append(item['text'])

    return text, targets

def list_text(text):
    
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text_sl = []
    # sum = 0

    for line in text:
        words = [w.lower() for w in line.strip().split() if (w not in stopwords and len(w)>=3)]
        # sum = sum + len(words)
        text_sl.append(words)
    
    # average number of words per whisper to find optimal value for 'window' in doc2vec
    # print(float(sum)/len(text_sl)) 
    return text_sl


def feature_vecs_doc(text_sl,test_sl):
    labeled_train_text = []
    
    for i,whisperWords in enumerate(text_sl):
        lso=LabeledSentence(words=whisperWords,tags=['TRAIN_'+str(i)])
        labeled_train_text.append(lso)

    labeled_test_text = []
    for i,whisperWords in enumerate(test_sl):
        lso=LabeledSentence(words=whisperWords,tags=['TEST_'+str(i)])
        labeled_test_text.append(lso)

    return labeled_train_text,labeled_test_text

def make_vecs(labeled_train_text,labeled_test_text,train_sl,test_sl):

    model = Doc2Vec(window = 8,min_count=10,size = 100) 
    sentences = labeled_train_text + labeled_test_text
    model.build_vocab(sentences)
    for i in range(5):
        print "Training iteration", i
        random.shuffle(sentences)
        model.train(sentences)

    train_vec=[]

    for i,fv in enumerate(train_sl):
        featureVec = model.docvecs['TRAIN_'+str(i)] 
        train_vec.append(featureVec)
   
    test_vec=[]

    for i,fv in enumerate(test_sl):
        featureVec = model.docvecs['TEST_'+str(i)]
        test_vec.append(featureVec)

    return train_vec,test_vec


def main():
    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')

    # Creating Lists of lists of words
    train_sl = list_text(text_train)
    test_sl = list_text(text_test)

    labeled_train_text,labeled_test_text = feature_vecs_doc(train_sl,test_sl)

    train_vec,test_vec = make_vecs(labeled_train_text,labeled_test_text,train_sl,test_sl)
    X = train_vec
    Y = targets_train
    
    lm  = sklearn.linear_model.LogisticRegression()
    lr_model = lm.fit(X,Y)

    predictionLM = lr_model.predict(test_vec)
    print 'macro f1 for linear model:', f1_score(targets_test, predictionLM, average='macro')
    # print 'weighted f1 for linear model:', f1_score(targets_test, predictionLM, average='weighted')

    nb  = sklearn.naive_bayes.BernoulliNB()
    nb_model = nb.fit(X,Y)

    predictionNB = nb_model.predict(test_vec)
    print 'macro f1 for Naive Bayes model:', f1_score(targets_test, predictionNB, average='macro')
    # print 'weighted f1 for Naive Bayes model:', f1_score(targets_test, predictionNB, average='weighted')



if __name__ == "__main__":
    main()
