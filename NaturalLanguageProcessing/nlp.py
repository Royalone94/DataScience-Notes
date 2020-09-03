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
    1.remove punc
    2. remove stopwords
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

head_func = messages['message'].head(5).apply(text_process)