#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

import matplotlib.pyplot as plt
np.random.seed(0)


# In[2]:


import os
print(os.listdir("../input"))


# In[3]:


n_df = pd.read_json('../input/News_Category_Dataset.json', lines=True)


# In[4]:


n_df['hds']=n_df['headline']+n_df['short_description']


# In[5]:


import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()


# In[6]:


def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    new1=''.join(nopunc)
    new2=[stemmer.stem(word) for word in new1]
    new3=''.join(new2)
    return[word for word in new3.split()if word.lower()not in stopwords.words('english') ]


# In[7]:


n_df['hds'].head()


# In[8]:


n_df['hds'].head(5).apply(text_process)


# In[9]:


train_text, test_text, ytrain, ytest = train_test_split(
    n_df['hds'], n_df['category'], random_state=52)


# In[10]:


get_ipython().run_cell_magic('time', '', "word_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='word',\n    token_pattern=r'\\w{1,}',\n    ngram_range=(1, 10))\nword_vectorizer.fit(train_text)")


# In[11]:


get_ipython().run_cell_magic('time', '', "char_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='char',\n    ngram_range=(1, 5))\nchar_vectorizer.fit(train_text)")


# In[12]:


get_ipython().run_cell_magic('time', '', 'X = hstack([word_vectorizer.transform(train_text), char_vectorizer.transform(train_text)])')


# In[13]:


from sklearn.linear_model import SGDClassifier
sgd_cls = SGDClassifier(max_iter=20)
sgd_cls.fit(X, ytrain)


# In[14]:


predict = sgd_cls.predict(hstack([word_vectorizer.transform(test_text), char_vectorizer.transform(test_text)]))


# In[15]:


acc = np.mean(ytest == predict)


# In[16]:


print('accuracy: {0:.3}'.format(acc))


# In[ ]:





# In[ ]:




