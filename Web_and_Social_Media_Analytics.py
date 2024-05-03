#!/usr/bin/env python
# coding: utf-8

# ## Load basic packages/modules

# In[1]:


import os
import numpy as np
import pandas as pd

import nltk
from nltk import corpus, tokenize
from nltk.corpus import stopwords

import re

from nltk.stem import PorterStemmer, WordNetLemmatizer, porter
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string

from sklearn.model_selection import train_test_split

# ML
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# DNN/MLP/ANN model


# In[2]:


pip install wordcloud


# ## Load the dataset

# In[3]:


hotstar = pd.read_csv("hotstar_reviews.csv")
hotstar.head()


# In[4]:


hotstar.shape


# In[5]:


hotstar.info()


# In[6]:


hotstar.Sentiment_Manual.value_counts()


# In[7]:


hotstar.Sentiment_Manual.value_counts()/hotstar.Sentiment_Manual.size*100


# In[8]:


# From above we can conclude that Sentiment_Manual is balanced


# In[9]:


hotstar.DataSource.value_counts()/hotstar.DataSource.size*100


# In[10]:


pd.pivot_table(hotstar, index='Sentiment_Manual', columns='DataSource',
              values='ID', aggfunc='count')/hotstar.DataSource.size*100


# In[11]:


hotstar.columns


# ## Data Cleaning

# In[12]:


review_data = hotstar[['Lower_Case_Reviews','DataSource','Sentiment_Manual']]
review_data.head()


# In[13]:


review_data.columns = ['Reviews','Source','Sentiment']
review_data.head()


# In[14]:


review_data.describe(include='object')


# In[15]:


review_data.Source.value_counts()


# In[16]:


review_data['Source'] = review_data['Source'].astype('category')
review_data['Source'] = review_data['Source'].cat.codes
review_data['Source'].value_counts()


# In[17]:


review_data.head()


# In[18]:


review_data.Sentiment.value_counts()


# In[19]:


review_data['Sentiment'] = review_data['Sentiment'].astype('category')
review_data['Sentiment'] = review_data['Sentiment'].cat.codes
review_data['Sentiment'].value_counts()


# In[20]:


import nltk
nltk.download('stopwords')


# In[21]:


punctuation = list(string.punctuation)

stop_words = stopwords.words('english')

re_pattern = """@[a-zA-Z0-9_:]+|b['"]rt|[\d]+[a-zA-Z_+='?]+[\d]+[\d]+|[a-zA-Z_*+=]+[\d]+[a-zA-Z_*+-=]+|[\d]+"""

re_patter = re_pattern + """|https:+[a-zA-Z0-9/._+-=]+|&amp;|rt"""

reviewText = [re.sub(pattern = re_pattern, string = text, repl="") 
               for text in review_data.Reviews.map(str).values]


# In[22]:


re_pattern


# In[23]:


print(reviewText[3])


# In[24]:


review_data_cleaned = []

for review in reviewText:
    stop_free = " ".join([txt for txt in review.lower().split() if txt not in stop_words])
    # stop_words - NLTK
    stop_free_1 = " ".join([txt for txt in stop_free.lower().split() if txt not in STOPWORDS])
    # STOPWORDS - WORDCLOUD
    puct_free = " ".join([txt for txt in stop_free_1.lower().split() if txt not in punctuation])
    review_data_cleaned.append(puct_free)


# In[25]:


# remove hashtags
review_data_cleaned_final = []

for rdcf in review_data_cleaned:
    final_words = rdcf.replace("#",'')
    review_data_cleaned_final.append(final_words)


# In[26]:


review_data_cleaned_final[7]


# ## Lemmatization

# In[27]:


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[28]:


wd = WordNetLemmatizer()

review_data_cleaned_final_output = []
for rdc in review_data_cleaned_final:
    clean_review = " ".join(wd.lemmatize(word) for word in rdc.split())
    review_data_cleaned_final_output.append(clean_review)


# In[29]:


review_data_cleaned_final_output


# In[30]:


review_data['clean_review'] = review_data_cleaned_final_output


# In[31]:


review_data.head()


# ## Split the data into training and test

# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(review_data['clean_review'],
                                                review_data['Sentiment'],test_size=0.2,
                                                   random_state=42,
                                                   stratify=review_data['Sentiment'])


# In[33]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# ## Feature Extraction

# In[34]:


# vectorize the text data using CountVectorizer

vectorizer = CountVectorizer(binary=True).fit(x_train)
x_train_vectorized = vectorizer.transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# In[35]:


x_train


# In[36]:


x_train_vectorized


# In[41]:


print(vectorizer.get_feature_names_out())


# In[42]:


pd.DataFrame(x_train_vectorized.toarray())


# In[43]:


pd.DataFrame(x_test_vectorized.toarray())


# ## Model Building

# In[44]:


from sklearn.naive_bayes import MultinomialNB


# In[45]:


naive_bayes_multi = MultinomialNB().fit(x_train_vectorized, y_train)


# In[46]:


print(naive_bayes_multi.score(x_train_vectorized, y_train))
print()
print(naive_bayes_multi.score(x_test_vectorized, y_test))


# In[49]:


predict_train = naive_bayes_multi.predict(x_train_vectorized)
predict_test = naive_bayes_multi.predict(x_test_vectorized)


# In[50]:


print(classification_report(y_train,predict_train ))
print()
print(classification_report(y_test,predict_test ))


# In[52]:


print(accuracy_score(y_train,predict_train ))
print()
print(accuracy_score(y_test,predict_test ))


# ## Random Forest Classifier

# In[53]:


rf_model = RandomForestClassifier().fit(x_train_vectorized, y_train)


# In[54]:


print(rf_model.score(x_train_vectorized, y_train))
print()
print(rf_model.score(x_test_vectorized, y_test))


# In[55]:


predict_train1 = rf_model.predict(x_train_vectorized)
predict_test1 = rf_model.predict(x_test_vectorized)


# In[57]:


print(classification_report(y_train,predict_train))
print()
print(classification_report(y_test,predict_test))


# In[59]:


print(accuracy_score(y_train,predict_train1 ))
print()
print(accuracy_score(y_test,predict_test1 ))


# ## K Fold method

# In[58]:


from sklearn.model_selection import cross_val_score
training_accuracy = cross_val_score(rf_model, x_train_vectorized, y_train, cv=10)
print(training_accuracy.mean())


# ## XGBoost Classifier

# In[60]:


xgboost = XGBClassifier().fit(x_train_vectorized, y_train)


# In[62]:


print(xgboost.score(x_train_vectorized, y_train))
print()
print(xgboost.score(x_test_vectorized, y_test))


# In[63]:


from sklearn.model_selection import cross_val_score
training_accuracy = cross_val_score(xgboost, x_train_vectorized, y_train, cv=10)
print(training_accuracy.mean())


# ## Feature Extraction - Term Frequency - Inverse Documents Frequency (TF-IDF)

# In[64]:


# vectorize the text data using TfidfVectorizer
tf_idf = TfidfVectorizer().fit(x_train)
x_train_tf_idf = tf_idf.transform(x_train)
x_test_tf_idf = tf_idf.transform(x_test)


# In[65]:


pd.DataFrame(x_test_tf_idf.toarray())


# In[66]:


x_test_tf_idf.toarray()


# In[67]:


xgboost = XGBClassifier().fit(x_train_tf_idf, y_train)


# In[68]:


print(xgboost.score(x_train_tf_idf, y_train))
print()
print(xgboost.score(x_test_tf_idf, y_test))


# ## Analysis using WordCloud

# In[69]:


review_data.head()


# In[70]:


review_data.Sentiment.value_counts()


# In[77]:


positive_review = review_data[review_data['Sentiment']==2]['clean_review']
negative_review = review_data[review_data['Sentiment']==0]['clean_review']
neutral_review = review_data[review_data['Sentiment']==1]['clean_review']


# In[78]:


positive_review.shape, negative_review.shape, neutral_review.shape


# In[80]:


import nltk
nltk.download('punkt')


# In[81]:


pos_data_token = [nltk.word_tokenize(rvw) for rvw in positive_review]
neg_data_token = [nltk.word_tokenize(rvw) for rvw in negative_review]
neu_data_token = [nltk.word_tokenize(rvw) for rvw in neutral_review]


# In[82]:


neu_data_token


# In[83]:


wordcloud_pos = WordCloud(background_color='blue', max_words=200, max_font_size=40,
                         scale=3, random_state=10).generate(str(pos_data_token))

wordcloud_neg = WordCloud(background_color='white', max_words=200, max_font_size=40,
                         scale=3, random_state=10).generate(str(neg_data_token))

wordcloud_neu = WordCloud(background_color='black', max_words=200, max_font_size=40,
                         scale=3, random_state=10).generate(str(neu_data_token))


# In[84]:


import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(12,12))
plt.axis('off')
plt.imshow(wordcloud_pos)
plt.title("Positive Review", size=20)
plt.show()


# In[85]:


import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(12,12))
plt.axis('off')
plt.imshow(wordcloud_neg)
plt.title("Negative Review", size=20)
plt.show()


# In[86]:


import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(12,12))
plt.axis('off')
plt.imshow(wordcloud_neu)
plt.title("Neutral Review", size=20)
plt.show()


# In[ ]:




