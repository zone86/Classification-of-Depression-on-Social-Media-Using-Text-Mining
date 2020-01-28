# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:11:38 2020

@author: cyrzon
"""

import pandas as pd
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import nltk
import pdb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import sklearn.metrics as metrics



Df = pd.read_csv('processed_data/all_possible_fallacies_manual_labelling_v1.csv')

data = Df.loc[:,['tweets']]
data= data.iloc[:, 0].tolist()

label = Df.loc[:,['fallacy']]
label = label.iloc[:, 0].tolist()
# label.fallacy = label.fallacy.astype(float) 


# create train test split
x_train = [] 
x_test = [] 
y_train = [] 
y_test = []

x_train, x_test, y_train, y_test = train_test_split(data, label, 
                                                    test_size = 0.3, 
                                                    random_state = 0, 
                                                    stratify = label)


vectorizer = CountVectorizer()
ngram_vectorizer = CountVectorizer(ngram_range = (2,3))
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

def bayes_CV(model, search, train, iterations, CV, score):
    opt = BayesSearchCV(model,
                        search,
                        n_iter = iterations,
                        cv = CV,
                        scoring = score)

    opt_result = opt.fit(train, y_train)
    
    #print("\n")
    #print("Best params: ", opt_result.best_params_)
    return opt_result.best_params_, opt_result.best_score_   

def grid_search(model, search, train, CV, score):
    opt = GridSearchCV(model,
                        search,
                        #n_iter = iterations,
                        cv = CV,
                        scoring = score)

    opt_result = opt.fit(train, y_train)
    
    print("\n")
    #print("Best params: ", opt_result.best_params_)
    #print("Best score: ", opt_result.best_score_)
    return opt_result.best_params_, opt_result.best_score_   

def nbTrain(vecType, tuning = False, tune_by = 'grid'):
    from sklearn.naive_bayes import MultinomialNB
    #start_timenb = time.time()
    
    train_features, test_features = getVecType(vecType)
       
    if tuning:
                       
        if tune_by == 'grid':
            
            grid_log_search = [{'penalty': ['l1'],
                               'solver': ['saga'],
                               'C': [0.01, 0.1, 1, 10, 100, 1000],                               
                               'max_iter': [100, 200, 300, 500, 800, 1000],
                               },
                                {'penalty': ['l2'],
                                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                                'C': [0.01, 0.1, 1, 10, 100, 1000],                               
                                'max_iter': [100, 200, 300, 500, 800, 1000]
                               }]
            
            grid_elastic_search = [{'alpha': [.01, .1, .2, .4, .5, .6, .7, .8, .9, .95, .99], 
                                    'max_iter': [100, 200, 300, 500, 800, 1000],
                                    'l1_ratio': [.01, .1, .2, 0.5, 0.6, 0.8, .9, .95, .99]}]
                                
        
            params = grid_search(model = m_nb, search = grid_log_search, 
                          train = train_features, 
                          CV = 3, score = 'roc_auc')
            
            elastic_params = grid_search(model = elastic_model, search = grid_elastic_search, 
                          train = train_features, 
                          CV = 3, score = 'roc_auc')
            
            if params[1] > elastic_params[1]:
                params = params[0]
                best_model = LogisticRegression(**params)
            else:
                params = elastic_params[0]
                best_model = ElasticNet(**params)
            
        elif tune_by == 'bayes':
            
            bayes_nb_search = [{
                    'alpha': Real(1e-6, 1, prior = 'log-uniform'),
                    'fit_prior': Categorical(['True','False'])
                    }]
            
            nb_model = MultinomialNB()
            params = bayes_CV(model = nb_model, search = bayes_nb_search, 
                              train = train_features, 
                              iterations = 20, CV = 3,
                              score = 'roc_auc')
                        
            
            best_model = MultinomialNB(**params[0])
        else:
            print('Need either bayes or grid as tune_by')
        
        
        
        best_model = best_model.fit(train_features, y_train)

        best_model_predictions = best_model.predict(test_features)
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, best_model_predictions , pos_label = 1)
        linear_score = format(metrics.auc(fpr, tpr))
        linear_score = float(linear_score)*100
        
        print("\n")
        print("Params : \n", params)
        print("\n")
        print("AUC : \n", linear_score,"%")
        #print(" Completion Speed", round((time.time() - start_timenb),5))
        print()
        
    else:
    
    
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

def logModTrain(vecType, tuning = False, tune_by = 'grid'):
    from sklearn.linear_model import LogisticRegression, ElasticNet
    
    #start_timenb = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    if tuning:
        log_model = LogisticRegression()
        elastic_model = ElasticNet()
        
        if tune_by == 'grid':
            
            grid_log_search = [{'penalty': ['l1'],
                               'solver': ['saga'],
                               'C': [0.01, 0.1, 1, 10, 100, 1000],                               
                               'max_iter': [100, 200, 300, 500, 800, 1000],
                               },
                                {'penalty': ['l2'],
                                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                                'C': [0.01, 0.1, 1, 10, 100, 1000],                               
                                'max_iter': [100, 200, 300, 500, 800, 1000]
                               }]
            
            grid_elastic_search = [{'alpha': [.01, .1, .2, .4, .5, .6, .7, .8, .9, .95, .99], 
                                    'max_iter': [100, 200, 300, 500, 800, 1000],
                                    'l1_ratio': [.01, .1, .2, 0.5, 0.6, 0.8, .9, .95, .99]}]
                                
        
            params = grid_search(model = log_model, search = grid_log_search, 
                          train = train_features, 
                          CV = 3, score = 'roc_auc')
            
            elastic_params = grid_search(model = elastic_model, search = grid_elastic_search, 
                          train = train_features, 
                          CV = 3, score = 'roc_auc')
            
            if params[1] > elastic_params[1]:
                params = params[0]
                best_model = LogisticRegression(**params)
            else:
                params = elastic_params[0]
                best_model = ElasticNet(**params)
            
        elif tune_by == 'bayes':
            
            bayes_logSearch = [{
                    'solver': ['saga'],
                    'C': Real(1e-6, 1000, prior = 'log-uniform'), 
                    'penalty': Categorical(['l1']),
                    'max_iter': Integer(100,1000)
                    },
                    {'solver': Categorical(['newton-cg', 'lbfgs', 'sag', 'saga']),
                     'C': Real(1e-6, .3, prior = 'log-uniform'), 
                     'penalty': Categorical(['l2']),
                     'max_iter': Integer(100,1000)
            }]
            
            bayes_elastic_search = [{'alpha': Real(.01, .99, prior = 'log-uniform'), 
                                    'max_iter': Integer(100, 1000),
                                    'l1_ratio': Real(.01, .99, prior = 'log-uniform')}]
            
            params = bayes_CV(model = log_model, search = bayes_logSearch, 
                              train = train_features, 
                              iterations = 20, CV = 3,
                              score = 'roc_auc')
            
            elastic_params = bayes_CV(model = elastic_model, search = bayes_elastic_search, 
                              train = train_features, 
                              iterations = 20, CV = 3,
                              score = 'roc_auc')
            
            if params[1] > elastic_params[1]:
                params = params[0]
                best_model = LogisticRegression(**params)
            else:
                params = elastic_params[0]
                best_model = ElasticNet(**params)
        else:
            print('Need either bayes or grid as tune_by')
        
        
        
        best_model = best_model.fit(train_features, y_train)

        best_model_predictions = best_model.predict(test_features)
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, best_model_predictions , pos_label = 1)
        linear_score = format(metrics.auc(fpr, tpr))
        linear_score = float(linear_score)*100
        
        print("\n")
        print("Params : \n", params)
        print("\n")
        print("AUC : \n", linear_score,"%")
        #print(" Completion Speed", round((time.time() - start_timenb),5))
        print()
        
    else:
            
        multi = LogisticRegression(solver = 'lbfgs')
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

    
def datree(vecType, tuning = False, tune_by = 'grid'):
    from sklearn import tree
    #start_timedt = time.time()
    
    train_features, test_features = getVecType(vecType)
    
    if tuning:
                       
        if tune_by == 'grid':
            
            grid_log_search = [{'penalty': ['l1'],
                               'solver': ['saga'],
                               'C': [0.01, 0.1, 1, 10, 100, 1000],                               
                               'max_iter': [100, 200, 300, 500, 800, 1000],
                               },
                                {'penalty': ['l2'],
                                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                                'C': [0.01, 0.1, 1, 10, 100, 1000],                               
                                'max_iter': [100, 200, 300, 500, 800, 1000]
                               }]
            
            grid_elastic_search = [{'alpha': [.01, .1, .2, .4, .5, .6, .7, .8, .9, .95, .99], 
                                    'max_iter': [100, 200, 300, 500, 800, 1000],
                                    'l1_ratio': [.01, .1, .2, 0.5, 0.6, 0.8, .9, .95, .99]}]
                                
        
            params = grid_search(model = m_nb, search = grid_log_search, 
                          train = train_features, 
                          CV = 3, score = 'roc_auc')
            
            elastic_params = grid_search(model = elastic_model, search = grid_elastic_search, 
                          train = train_features, 
                          CV = 3, score = 'roc_auc')
            
            if params[1] > elastic_params[1]:
                params = params[0]
                best_model = LogisticRegression(**params)
            else:
                params = elastic_params[0]
                best_model = ElasticNet(**params)
            
        elif tune_by == 'bayes':
            
            bayes_dtree_search = [{
                    'criterion': Categorical(['gini', 'entropy']), 
                    'splitter': Categorical(['best', 'random']),
                    'min_samples_split': Integer(10, 50),
                    'min_samples_leaf': Integer(1, 50),
                    'max_features': (['auto', 'sqrt', 'log2'])
                    
                    }]
            
            dtree = tree.DecisionTreeClassifier()
            params = bayes_CV(model = dtree, search = bayes_dtree_search, 
                              train = train_features, 
                              iterations = 20, CV = 3,
                              score = 'roc_auc')
                        
            
            best_model = tree.DecisionTreeClassifier(**params[0])
        else:
            print('Need either bayes or grid as tune_by')
        
        
        
        best_model = best_model.fit(train_features, y_train)

        best_model_predictions = best_model.predict(test_features)
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, best_model_predictions , pos_label = 1)
        linear_score = format(metrics.auc(fpr, tpr))
        linear_score = float(linear_score)*100
        
        print("\n")
        print("Params : \n", params)
        print("\n")
        print("AUC : \n", linear_score,"%")
        #print(" Completion Speed", round((time.time() - start_timenb),5))
        print()
        
    else:
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
    
def mlp(vecType):
    from sklearn.neural_network import MLPClassifier
    #start_timerf = time.time()
    
    train_features, test_features = getVecType(vecType)
       
    mlp = MLPClassifier(random_state=0)
    
    
    mlp = mlp.fit(train_features, [int(i) for i in y_train])
    prediction4 = mlp.predict(test_features)
    mmm, ppp, thresholds = metrics.roc_curve(y_test, prediction4, pos_label=1)
    kn = format(metrics.auc(mmm, ppp))
    kn = float(kn)*100
    print("MLP AUC : \n", kn, "%")
    #print(" Completion Speed", round((time.time() - start_timerf),5))
    print()
    print()

logModTrain(vecType = 'BOW', tuning = False, tune_by = 'bayes')
nbTrain(vecType = 'tfidf', tuning = False, tune_by = 'bayes')
datree(vecType = 'tfidf')
Tsvm(vecType = 'tfidf')
knN(vecType = 'tfidf')
RanFo(vecType = 'BOW')
mlp(vecType= 'BOW')