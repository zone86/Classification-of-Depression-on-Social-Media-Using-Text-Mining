#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on the day we all start to love our self.

@author: Cyrzon
"""

import json
import pandas as pd
import time
import numpy as np
import re
from nltk.corpus import stopwords
import itertools
import matplotlib.pyplot as plt 
#from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.metrics import roc_auc_score



def retrieveTweet(data_url):

    tweets_data_path = data_url
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

def retrieveProcessedData(Pdata_url):
    sent = pd.read_excel(Pdata_url)
    for i in range(len(tweets_data)):
        if tweets_data[i]['id']==sent['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sent['sentiment'][i])
            
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

# function to clean text. models perform better with stop words set to false
def clean_text(text, remove_stopwords = False):
    
    # convert to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    #text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'[_"\-;%()|+&=*%:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    prefix = 'rt'
    if text.startswith(prefix):
        return text[len(prefix):]
    return text      

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
            
    return text

            
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
            
            
def getVecType(vecType):
    
    if vecType == 'BOW':
        train_features = vectorizer.fit_transform(x_train)
        test_features = vectorizer.transform(x_test)
    elif vecType == 'ngram':
        train_features = ngram_vectorizer.fit_transform(x_train)
        test_features = ngram_vectorizer.transform(x_test)
    elif vecType == 'tfidf':
        train_features = tfidf_vectorizer.fit_transform(x_train)
        test_features = tfidf_vectorizer.transform(x_test)
        
    return train_features, test_features


def nbTrain(vecType):
    from sklearn.naive_bayes import MultinomialNB
    start_timenb = time.time()
    
    train_features, test_features = getVecType(vecType)
       
    nb = MultinomialNB()
    nb.fit(train_features, [int(r) for r in y_train])
    
    predictions = nb.predict(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    nbscore = format(metrics.auc(fpr, tpr))
    nbscore = float(nbscore)*100
    
    print("\n")
    print("Naive Bayes  AUC : \n", nbscore,"%")
    print(" Completion Speed", round((time.time() - start_timenb),5))
    print()
    
# =============================================================================
#     nb_matrix = confusion_matrix(y_test, predictions)
#     plt.figure()
#     plot_confusion_matrix(nb_matrix, classes=[-1,0,1], title='Confusion matrix For NB classifier')
# =============================================================================
    
#    test_try= vectorizer.transform(["Lets help those in need, fight anxiety and bring happiness"])
#    test_try2= vectorizer.transform(["Dont look down at people with anxiety rather give love and respect to all. shout! Equality."])
#    predictr = nb.predict(test_try)
#    predictt = nb.predict(test_try2)
    
    
#    print(predictr)
#    print(predictt)

def logModTrain(vecType):
    from sklearn.linear_model import LogisticRegression
    start_timenb = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    multi = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial' )
    multi.fit(train_features, [int(r) for r in y_train])
    
    multi_predictions = multi.predict(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, multi_predictions, pos_label=1)
    linear_score = format(metrics.auc(fpr, tpr))
    linear_score = float(linear_score)*100
    
    print("\n")
    print("Multinomial  AUC : \n", linear_score,"%")
    print(" Completion Speed", round((time.time() - start_timenb),5))
    print()

    
# =============================================================================
#     linear_matrix = confusion_matrix(y_test, multi_predictions)
#     plt.figure()
#     plot_confusion_matrix(linear_matrix, classes=[-1,0,1], title='Confusion matrix For multinomial classifier')
# =============================================================================
    

#    test_try= vectorizer.transform(["Lets help those in need, fight anxiety and bring happiness"])
#    test_try2= vectorizer.transform(["Dont look down at people with anxiety rather give love and respect to all. shout! Equality."])
#    predictr = nb.predict(test_try)
#    predictt = nb.predict(test_try2)
        
#    print(predictr)
#    print(predictt)

    
def datree(vecType):
    from sklearn import tree
    start_timedt = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    dtree = tree.DecisionTreeClassifier()
    
    dtree = dtree.fit(train_features, [int(r) for r in y_train])
    
    prediction1 = dtree.predict(test_features)
    ddd, ttt, thresholds = metrics.roc_curve(y_test, prediction1, pos_label=1)
    dtreescore = format(metrics.auc(ddd, ttt))
    dtreescore = float(dtreescore)*100
    print("Decision tree AUC : \n", dtreescore, "%")
    print(" Completion Speed", round((time.time() - start_timedt),5))
    print()


def Tsvm(vecType):
    from sklearn.svm import SVC
    start_timesvm = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    svc = SVC()
    
    svc = svc.fit(train_features, [int(r) for r in y_train])
    prediction2 = svc.predict(test_features)
    sss, vvv, thresholds = metrics.roc_curve(y_test, prediction2, pos_label=1)
    svc = format(metrics.auc(sss, vvv))
    svc = float(svc)*100
    print("Support vector machine AUC : \n", svc, "%")
    print(" Completion Speed", round((time.time() - start_timesvm),5))
    print()

def knN(vecType):
    from sklearn.neighbors import KNeighborsClassifier
    start_timekn = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    kn = KNeighborsClassifier(n_neighbors=2, 
                              algorithm='kd_tree', 
                              leaf_size = 20000)
        
    kn = kn.fit(train_features, [int(i) for i in y_train])
    prediction3 = kn.predict(test_features)
    kkk, nnn, thresholds = metrics.roc_curve(y_test, prediction3, pos_label=1)
    kn = format(metrics.auc(kkk, nnn))
    kn = float(kn)*100
    
    print("Kneighborsclassifier AUC : \n", kn, "%")
    print(" Completion Speed", round((time.time() - start_timekn),5))
    print()

def RanFo(vecType):
    from sklearn.ensemble import RandomForestClassifier
    start_timerf = time.time()
    
    train_features, test_features = getVecType(vecType)
       
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    
    
    rf = rf.fit(train_features, [int(i) for i in y_train])
    prediction4 = rf.predict(test_features)
    rrr, fff, thresholds = metrics.roc_curve(y_test, prediction4, pos_label=1)
    kn = format(metrics.auc(rrr, fff))
    kn = float(kn)*100
    print("Random Forest AUC : \n", kn, "%")
    print(" Completion Speed", round((time.time() - start_timerf),5))
    print()
    print()




vectorizer = CountVectorizer()
ngram_vectorizer = CountVectorizer(ngram_range = (2,2))
tfidf_vectorizer = TfidfVectorizer(min_df = 0.03, max_df = 0.97, 
                                   ngram_range = (2,2))

tweets_data = []
x = []
y = []
x_train = [] 
x_test = [] 
y_train = [] 
y_test = []


retrieveTweet('data/tweetdata.txt')  
retrieveProcessedData('processed_data/output.xlsx')
clean_text(x, remove_stopwords = True)

cleaned_tweets = []

for i in range(0,len(x)):
    string = clean_text(x[i], remove_stopwords = False)
    cleaned_tweets.append(string)


x_train, x_test, y_train, y_test = train_test_split(cleaned_tweets, y, 
                                                    test_size=0.3, 
                                                    random_state=0, 
                                                    stratify=y)
    
        
logModTrain(vecType = 'ngram')
nbTrain(vecType = 'ngram')
datree(vecType = 'ngram')
Tsvm(vecType = 'ngram')
knN(vecType = 'ngram')
RanFo(vecType = 'ngram')
    
# =============================================================================
# def datreeINPUT(inputtweet):
#     from sklearn import tree
#     train_featurestree = vectorizer.fit_transform(x)
#     dtree = tree.DecisionTreeClassifier()
#     
#     dtree = dtree.fit(train_featurestree, [int(r) for r in y])
#     
#     
#     inputdtree= vectorizer.transform([inputtweet])
#     predictt = dtree.predict(inputdtree)
#     
#     if predictt == 1:
#         predictt = "Positive"
#     elif predictt == 0:
#         predictt = "Neutral"
#     elif predictt == -1:
#         predictt = "Negative"
#     else:
#         print("Nothing")
#     
#     print("\n*****************")
#     print(predictt)
#     print("*****************")
# 
# runall()
# 
# print("Input your tweet : ")
# inputtweet = input()
# 
# datreeINPUT(inputtweet)
# 
# =============================================================================
