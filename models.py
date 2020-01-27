# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:11:38 2020

@author: cyrzon
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

# create train test split
x_train = [] 
x_test = [] 
y_train = [] 
y_test = []

x_train, x_test, y_train, y_test = train_test_split(Df1, sentiment, 
                                                    test_size=0.3, 
                                                    random_state=0, 
                                                    stratify=sentiment)


vectorizer = CountVectorizer()
ngram_vectorizer = CountVectorizer(ngram_range = (2,2))
tfidf_vectorizer = TfidfVectorizer(min_df = 0.03, max_df = 0.97, 
                                   ngram_range = (2,2))


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
    #start_timenb = time.time()
    
    train_features, test_features = getVecType(vecType)
       
    nb = MultinomialNB()
    nb.fit(train_features, [int(r) for r in y_train])
    
    predictions = nb.predict(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    nbscore = format(metrics.auc(fpr, tpr))
    nbscore = float(nbscore)*100
    
    print("\n")
    print("Naive Bayes  AUC : \n", nbscore,"%")
    #print(" Completion Speed", round((time.time() - start_timenb),5))
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
    #start_timenb = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    multi = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial' )
    multi.fit(train_features, [int(r) for r in y_train])
    
    multi_predictions = multi.predict(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, multi_predictions, pos_label=1)
    linear_score = format(metrics.auc(fpr, tpr))
    linear_score = float(linear_score)*100
    
    print("\n")
    print("Multinomial  AUC : \n", linear_score,"%")
    #print(" Completion Speed", round((time.time() - start_timenb),5))
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
    #start_timedt = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    dtree = tree.DecisionTreeClassifier()
    
    dtree = dtree.fit(train_features, [int(r) for r in y_train])
    
    prediction1 = dtree.predict(test_features)
    ddd, ttt, thresholds = metrics.roc_curve(y_test, prediction1, pos_label=1)
    dtreescore = format(metrics.auc(ddd, ttt))
    dtreescore = float(dtreescore)*100
    print("Decision tree AUC : \n", dtreescore, "%")
    #print(" Completion Speed", round((time.time() - start_timedt),5))
    print()


def Tsvm(vecType):
    from sklearn.svm import SVC
    #start_timesvm = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    svc = SVC()
    
    svc = svc.fit(train_features, [int(r) for r in y_train])
    prediction2 = svc.predict(test_features)
    sss, vvv, thresholds = metrics.roc_curve(y_test, prediction2, pos_label=1)
    svc = format(metrics.auc(sss, vvv))
    svc = float(svc)*100
    print("Support vector machine AUC : \n", svc, "%")
    #print(" Completion Speed", round((time.time() - start_timesvm),5))
    print()

def knN(vecType):
    from sklearn.neighbors import KNeighborsClassifier
    #start_timekn = time.time()
    
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
    #print(" Completion Speed", round((time.time() - start_timekn),5))
    print()

def RanFo(vecType):
    from sklearn.ensemble import RandomForestClassifier
    #start_timerf = time.time()
    
    train_features, test_features = getVecType(vecType)
       
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    
    
    rf = rf.fit(train_features, [int(i) for i in y_train])
    prediction4 = rf.predict(test_features)
    rrr, fff, thresholds = metrics.roc_curve(y_test, prediction4, pos_label=1)
    kn = format(metrics.auc(rrr, fff))
    kn = float(kn)*100
    print("Random Forest AUC : \n", kn, "%")
    #print(" Completion Speed", round((time.time() - start_timerf),5))
    print()
    print()

logModTrain(vecType = 'BOW')
nbTrain(vecType = 'ngram')
datree(vecType = 'BOW')
Tsvm(vecType = 'BOW')
knN(vecType = 'BOW')
RanFo(vecType = 'BOW')
