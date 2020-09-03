# -*- coding: utf-8 -*-

# Natural Language Processing 

import nltk
#nltk.download_shell()
messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))
print(messages[50])

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    
    
import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
describe = messages.describe()
descibe2 = messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)
head = messages.head()

import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot.hist(bins=50)

long = messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

import string
mess = 'Sample message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]
from nltk.corpus import stopwords
stopwords.words('english')
nopunc = ''.join(nopunc)
nopunc.split()
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def text_process(mess):
    """ 
    1. remove punc
    2. remove stopwords
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

head_func = messages['message'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
mess4 = messages['message'][3]
bow4 = bow_transformer.transform([mess4])
bow_transformer.get_feature_names()[9554]

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of the Sparse Matrix: ', messages_bow.shape)
messages_bow.nnz

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB().fit(messages_tfidf, messages['label'])
spam_detect.predict(tfidf4)[0]
messages['label'][3]

all_pred = spam_detect.predict(messages_tfidf)

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3, random_state=101)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())    
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
