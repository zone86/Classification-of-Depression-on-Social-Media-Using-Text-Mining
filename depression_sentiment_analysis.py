#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cyrzon
"""

import json
import pandas as pd
import numpy as np
import re
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import nltk
import itertools
import pdb
import csv
import time
import math

#import string
import preprocessor as p

# =============================================================================
# import fallacies
# 
# all_fallacies = fallacies.all_fallacies
# 
# =============================================================================
#lemmatizer = LancasterStemmer()

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


# function to clean text. models perform better with stop words set to false
def clean_text(text, replace_contractions = True):
    
    # convert to lower case
    text = text.lower()
    
    #lemmatized = []
    #for word in text:
    #    lemmatized.append(lemmatizer.lemmatize(word))
    
    
    # Replace contractions with their longer forms 
    if replace_contractions:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
        
        text = text.split()
        new_text_2 = []
        for word_2 in text:
            if word_2 in contractions_2:
                new_text_2.append(contractions_2[word_2])
            else:
                new_text_2.append(word_2)
        text = " ".join(new_text_2)
    
    #pdb.set_trace()
    
    # Format words and remove unwanted characters
    text = re.sub('(http\S+)|(https\S+)', '', text) # remove URLs
    text = re.sub(r'@[^\s]+', '', text) # remove re-tweets
    
    if text.startswith('rt '): # remove re-tweets
        text = text[len('rt ')+1:]
        
    text = re.sub(r'(.)\1+', r'\1\1', text) # remove repeating characters
    
    text = re.sub(r'#([^\s]+)', r'\1', text) # remove hashtags
    text = re.sub(r'[_"\-;%()|+&=~*%:]', '', text) # remove characters
    
    text = re.sub(r'[^\x00-\x7F]+',' ', text) #remove emojis from tweet
    text = emoji_pattern.sub(r'', text)#filter using NLTK library append it to a string
    
    text = re.sub("[<@*&?].*[>@*&?]", "", text) # remove everything between < and >
    
    # Custom removals
    text = re.sub(r'&amp', '', text)
    text = re.sub(r' amp ', ' and ', text)
    text = re.sub(r' idk ', ' i do not know ', text)
    text = re.sub(r' idc ', ' i do not care ', text)
    text = re.sub(r' ppl', ' people ', text)
    text = re.sub(r' u ', ' you ', text)
    text = re.sub(r' r ', ' are ', text)
    text = re.sub(r' bc ', ' because ', text)
    text = re.sub(r' ur ', ' you are ', text)
    text = re.sub(r' w ', ' with ', text)
    text = re.sub(r' rn ', ' right now ', text)
    text = re.sub(r' cuz ', ' because ', text)
    text = re.sub(r' cos ', ' because ', text)
    text = re.sub(r' coz ', ' because ', text)
    text = re.sub(r' dunno ', ' do not know ', text)
    text = re.sub(r' gonna ', ' going to ', text)
    text = re.sub(r' tho ', ' although ', text)
    text = re.sub(r' thru ', ' through ', text)
    text = re.sub(r' plz ', ' please ', text)
    text = re.sub(r' jk ', ' joking ', text)
    text = re.sub(r'yup', 'yes', text)
    text = re.sub(r' \'em',' them', text)
    text = re.sub(r' til ', 'until ', text)
    text = re.sub(r' cnt ', ' can not ', text)
    text = re.sub(r' frm ', ' from ', text)
        
    return text

    

emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

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
"y'all": "you all",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}  
   
contractions_2 = { 
"aint": "am not",
"arent": "are not",
"cant": "cannot",
"cantve": "cannot have",
"cause": "because",
"couldve": "could have",
"couldnt": "could not",
"couldntve": "could not have",
"didnt": "did not",
"doesnt": "does not",
"dont": "do not",
"hadnt": "had not",
"hadntve": "had not have",
"hasnt": "has not",
"havent": "have not",
#"hed": "he would",
"hedve": "he would have",
#"hell": "he will",
"hes": "he is",
"howd": "how did",
"howll": "how will",
"hows": "how is",
"id": "i would",
"ill": "i will",
"im": "i am",
"ive": "i have",
"isnt": "is not",
"itd": "it would",
"itll": "it will",
"its": "it is",
"lets": "let us",
"maam": "madam",
"maynt": "may not",
"mightve": "might have",
"mightnt": "might not",
"mustve": "must have",
"mustnt": "must not",
"neednt": "need not",
"oughtnt": "ought not",
"shant": "shall not",
"shant": "shall not",
#"shed": "she would",
#"shell": "she will",
"shes": "she is",
"shouldve": "should have",
"shouldnt": "should not",
"thatd": "that would",
"thats": "that is",
"thered": "there had",
"theres": "there is",
"theyd": "they would",
"theyll": "they will",
"theyre": "they are",
"theyve": "they have",
"wasnt": "was not",
#"wed": "we would",
#"well": "we will",
#"were": "we are",
"weve": "we have",
"werent": "were not",
"whatll": "what will",
"whatre": "what are",
"whats": "what is",
"whatve": "what have",
"whered": "where did",
"wheres": "where is",
"wholl": "who will",
"whos": "who is",
"wont": "will not",
"wouldnt": "would not",
"yall": "you all",
"youd": "you would",
"youll": "you will",
"youre": "you are"
}          
            
##########################################################################

#retrieveTweet('data/tweetdata.txt')  
controlTweets = open('data/controlTweets.csv', encoding = 'ISO-8859-1', errors = 'ignore')
controlTweets_read = pd.read_csv(controlTweets, index_col=False)

# source preprocessor
#get_csv_data('data/controlTweets.csv')
#readdict('data/dictionary.tsv')

cleaned_tweets = []

for i in range(0,len(controlTweets_read)):
    string = p.clean(controlTweets_read.iloc[i,1]) # clean tweets with module
    cleaned_tweets.append(string)

cleaned_tweets_manual = []

for i in range(0,len(cleaned_tweets)):
    string = clean_text(cleaned_tweets[i]) # custom function to clean what the module missed
    cleaned_tweets_manual.append(string)
    
sentiment = pd.read_excel('processed_data/output.xlsx')
text = pd.DataFrame(cleaned_tweets_manual, columns = ['tweets'])
bind_tweet_sent = pd.concat([sentiment.reset_index(drop=True), text],
                            axis = 1)
bind_tweet_sent.to_csv('processed_data/tweets_sentiments.csv', index=False)

################################################################################

# add cognitive fallacies

tweets_sentiments = pd.read_csv('processed_data/tweets_sentiments.csv')
#retrieveProcessedData('processed_data/output.xlsx')

#all_fallacies_stack = all_fallacies.melt(id_vars)
categorized_fallacies = np.zeros((len(tweets_sentiments),all_fallacies.shape[1]), dtype="float32")

clean_tweets_length = range(0,len(tweets_sentiments))

all_fallacies_length = range(0,all_fallacies.shape[0])
all_fallacies_width = range(0,all_fallacies.shape[1])

length = all_fallacies.shape[0]
width = all_fallacies.shape[1]

### experimenting with faster method ???

lengths = np.zeros((length,width), dtype="float32")

for column in all_fallacies_width:    
        
    fallacy_words = all_fallacies.iloc[:,column]
        
    for fallacy in all_fallacies_length:
        fallacy_words_length  = len(fallacy_words[fallacy].split() )
        lengths[fallacy, column] = fallacy_words_length
 
lengths_df = pd.DataFrame(lengths)



for tweet in clean_tweets_length:
    l = list(lengths_df.itertuples(index=False, name=None))
    for i in l:
        #start_time = time.time()
        pdb.set_trace()
        fallacy_words = all_fallacies.iloc[:,index]
    
        for fallacy in range(0,all_fallacies_length):
            fallacy_words_length = len(fallacy_words[fallacy].split())
            
            if fallacy_words_length == 1:
                a = fallacy_words[fallacy]
                b = cleaned_tweets_manual[tweet].split()
                if a in b:
                    categorized_fallacies[tweet,column] += 1
                else:
                    categorized_fallacies[tweet,column] += 0
            else:
                words = fallacy_words[fallacy].split()
                string = cleaned_tweets_manual[tweet]
                
                if all(x in string for x in words):
                    categorized_fallacies[tweet,column] += 1
                else:
                    categorized_fallacies[tweet,column] += 0
    print("Processing time: ", round((time.time() - start_time),8), "Seconds \n\n")


# with map and list comprehension???
l = list(itertools.product(all_fallacies_length, all_fallacies_width))
result = [len(all_fallacies.iloc[i,x].split()) for i, x in enumerate(l)]

tuples = list(itertools.product(all_fallacies_width, all_fallacies_length))
result = list(map(lambda m: all_fallacies.iloc[m[0],m[1]].split(), tuples))
print(result)

    
##################################################################################
### DONT TOUCH - WOKRING!!! ### ### ### 

for tweet in clean_tweets_length:
    #pdb.set_trace()
    if pd.isnull(tweets_sentiments.iloc[tweet]['tweets']):
        continue
    
    for column in all_fallacies_width:            
        fallacy_words = all_fallacies.iloc[:,column]
        
        for fallacy in all_fallacies_length:
            fallacy_words_length  = len(fallacy_words[fallacy].split() )
 
            if fallacy_words_length == 1:
                #pdb.set_trace()
                a = fallacy_words[fallacy]
                b = tweets_sentiments.iloc[tweet]['tweets'].split()
                #pdb.set_trace()       
                                     
                if a in b:
                    #pdb.set_trace()
                    categorized_fallacies[tweet,column] += 1
                else:
                    #pdb.set_trace()
                    categorized_fallacies[tweet,column] += 0                
                
            else:
                
                words = fallacy_words[fallacy].split()
                string = tweets_sentiments.iloc[tweet]['tweets']
                
                if all(x in string for x in words):
                    #pdb.set_trace()
                    categorized_fallacies[tweet,column] += 1
                else:
                    categorized_fallacies[tweet,column] += 0



scored_fallacies = pd.DataFrame(data = categorized_fallacies, columns = all_fallacies.columns)
bind_dat = pd.concat([tweets_sentiments.reset_index(drop=True), 
                      #sentiment, 
                      scored_fallacies], axis = 1)


bind_dat.to_csv('processed_data/categorized_fallacies.csv', index=False)

#####################################################################################
#####################################################################################

Df = pd.read_csv('processed_data/categorized_fallacies.csv')


total = Df.iloc[:,2:].sum(axis=1)
total_Df = pd.DataFrame({"total": total})

Df1 = pd.concat([Df.reset_index(drop=True), total_Df], axis = 1)

Df1 = Df1.sort_values('tweets', ascending=False)
Df1 = Df1.drop_duplicates(subset='tweets', keep='first')

idx = Df1.index[(Df1['total'] > 0) 
#& (Df1['sentiment'] < 0)
]

all_possible_fallacies = Df1.loc[idx]
negative_possible_fallacies = Df1.loc[idx]
    
all_possible_fallacies.to_csv('processed_data/all_possible_fallacies.csv', index=False)
negative_possible_fallacies.to_csv('processed_data/negative_possible_fallacies.csv', index=False)