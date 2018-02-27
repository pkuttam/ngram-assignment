#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:52:55 2018

@author: pk.uttam
"""
# 
# In[]
import numpy as np
import itertools

# In[]
from nltk.corpus import gutenberg
from nltk.corpus import brown

# In[]

# For only gutenberg as train data-> select select_train = "g"
# For only brown as train data-> select select_train = "b"
# For gutenberg and brown as train data-> select select_train = "gb"
select_train = "gb"
# For gutenberg as train data-> select select_train = "g"
# For brown as train data-> select select_train = "b"
select_test = "b"


# In[]

sentnce_g = gutenberg.sents()
sentnce_b = brown.sents()

# sentence normalization with start S and end E tag augmentation

# for gutenberg dataset
sents_g = [];

for sents in sentnce_g:
    sent_lower = [x.lower() for x in sents ] 
    sents_g.append(['S','S']+sent_lower+['E'])


num_sents_g = len(sents_g)

num_train_g = int(num_sents_g*0.8)
num_test_g = num_sents_g-num_train_g

train_sents_g = sents_g[0:num_train_g]
test_sents_g = sents_g[num_train_g:num_sents_g]

# for brown dataset

sents_b = [];

for sents in sentnce_b:
    sent_lower = [x.lower() for x in sents ] 
    sents_b.append(['S','S']+sent_lower+['E'])


num_sents_b = len(sents_b)

num_train_b = int(num_sents_b*0.8)
num_test_b = num_sents_b-num_train_b

train_sents_b = sents_b[0:num_train_b]
test_sents_b = sents_b[num_train_b:num_sents_b]

# deleting for memory release

del sents_g
del sents_b
del gutenberg
del brown
del sentnce_g
del sentnce_b

# gutenberg and brown dataset
train_sents_gb = train_sents_b + train_sents_g
test_sents_gb = test_sents_b + test_sents_g

if select_train =="g":
    train_sents_g = train_sents_g
elif select_train=="b":
    train_sents_g = train_sents_b
elif select_train=="gb":
    train_sents_g = train_sents_gb
else:
    print("wrong dataset selected. Please specify- g,b or gb")
    
if select_test =="g":
    test_sents_g = test_sents_g
elif select_test=="b":
    test_sents_g = test_sents_b
else:
    print("wrong dataset as test selected. Please specify- g or b")


# In[]
train_words_g = list(itertools.chain.from_iterable(train_sents_g))
ngram_1_g = {}

# uni-gram freq count
for word in train_words_g:
    key = word
    if key in ngram_1_g:
        ngram_1_g[key]=ngram_1_g[key] +1 
    elif key not in ngram_1_g:
        ngram_1_g[key]=1

# In[]
## UNK tag augmentation
max_freq = 1

key_ngram_1_g = list(ngram_1_g.keys())

ngram_1_g_aug = dict(ngram_1_g)

ngram_1_g_aug['UNK'] = 0

for word in key_ngram_1_g:
    if ngram_1_g[word]<=max_freq:
        #print(ngram_1_g[word])
        ngram_1_g_aug['UNK']=ngram_1_g_aug['UNK']+ngram_1_g[word]
        del ngram_1_g_aug[word]


# In[]
#  sentence augmentation/normalization

train_sents_g_aug = []
for sents in train_sents_g:
    sent_chnge =[ x if x in ngram_1_g_aug else 'UNK' for x in sents] 
    train_sents_g_aug.append(sent_chnge)



# In[]
# Bigram count 
ngram_2_g_aug={}
for sents in train_sents_g_aug:
    length = len(sents)
    for i in range(0,length-1):
        key = (sents[i],sents[i+1])
        if key in ngram_2_g_aug:
            ngram_2_g_aug[key] = ngram_2_g_aug[key]+1
        else:
            ngram_2_g_aug[key] = 1

# In[]
# trigram count 
ngram_3_g_aug={}
for sents in train_sents_g_aug:
    length = len(sents)
    for i in range(0,length-2):
        key = (sents[i],sents[i+1],sents[i+2])
        if key in ngram_3_g_aug:
            ngram_3_g_aug[key] = ngram_3_g_aug[key]+1
        else:
            ngram_3_g_aug[key] = 1


# In[]

# All existing key in dictionary . Those which doesn't exit and exist keynser smoothing
keys_ngram_1_g_aug =list( ngram_1_g_aug.keys())

keys_ngram_2_g_aug =list( ngram_2_g_aug.keys())

keys_ngram_3_g_aug =list( ngram_3_g_aug.keys())




# In[]

# test data  sentence augmentation/normalization

test_sents_g_aug = []
for sents in test_sents_g:
    sent_chnge =[ x if x in ngram_1_g_aug else 'UNK' for x in sents] 
    test_sents_g_aug.append(sent_chnge)

# In[]

#KN for unigram
d = 0.75

# unigram continuation - All the unique count of word before w/ total biagram count
P_CN_1_g_nemu = {} # all unique count of w at place 2 n bigram dictionary
P_CN_1_g = {}
num_unique_bigrams = len(keys_ngram_1_g_aug)

for key2 in keys_ngram_2_g_aug:
    if key2[1] in P_CN_1_g_nemu:
        P_CN_1_g_nemu[key2[1]] = P_CN_1_g_nemu[key2[1]] + 1
    else:
        P_CN_1_g_nemu[key2[1]] = 1

for key in P_CN_1_g_nemu:
    P_CN_1_g[key] =   P_CN_1_g_nemu[key]*1.0/num_unique_bigrams
      


# unigram lamda
lamda_1_g_nemu = {}
for key2 in keys_ngram_2_g_aug:
    if key2[0] in lamda_1_g_nemu:
        lamda_1_g_nemu[key2[0]] = lamda_1_g_nemu[key2[0]] + 1
    else:
        lamda_1_g_nemu[key2[0]] =  1

lamda_1_g={}
for key in lamda_1_g_nemu:
    lamda_1_g[key] = d*lamda_1_g_nemu[key]*1.0/ngram_1_g_aug[key]


# probability KN
P_KN_2_g = {}
for key2 in keys_ngram_2_g_aug:
    P_KN_2_g[key2] = (ngram_2_g_aug[key2] -d)/ngram_1_g_aug[key2[0]] + lamda_1_g[key2[0]]*P_CN_1_g[key2[1]]


# In[]
# unigram perplexity
logP_1_g = 0;
comb_total = 0;
for sents in test_sents_g_aug:
    length = len(sents)
    for i in range(0,length):
        comb_total = comb_total+1
        key = sents[i]
        prob = P_CN_1_g[key]
        logP_1_g = logP_1_g + np.log(prob)



ppxty_1_g =np.exp(-logP_1_g/comb_total)
#print("unigram perplexity = "+ str(np.floor(ppxty_1_g)))



# In[]
    
# bigram perplexity on gutenberg test data
logP_2_g = 0;
comb_total = 0;
for sents in test_sents_g_aug:
    length = len(sents)
    for i in range(0,length-1):
        comb_total = comb_total+1
        key = (sents[i],sents[i+1])
        if key in ngram_2_g_aug:
            prob = (ngram_2_g_aug[key] -d)/ngram_1_g_aug[key[0]] + lamda_1_g[key[0]]*P_CN_1_g[key[1]]
            logP_2_g = logP_2_g + np.log(prob)
        else:
            prob = lamda_1_g[key[0]]*P_CN_1_g[key[1]]
            logP_2_g = logP_2_g + np.log(prob)

ppxty_2_g =np.exp(-logP_2_g/comb_total)
print("bigram perplexity = "+ str(np.floor(ppxty_2_g)))



# In[]

# KN for trigram
# bigram lamda
lamda_2_g_nemu = {}
for key3 in keys_ngram_3_g_aug:
    if key3[0:2] in lamda_2_g_nemu:
        lamda_2_g_nemu[key3[0:2]] = lamda_2_g_nemu[key3[0:2]] + 1
    else:
        lamda_2_g_nemu[key3[0:2]] =  1

lamda_2_g={}
for key2 in lamda_2_g_nemu:
    lamda_2_g[key2] = d*lamda_2_g_nemu[key2]*1.0/ngram_2_g_aug[key2]


def P_KN_3_g(key3):
    if key3 in ngram_3_g_aug:
        prob = (ngram_3_g_aug[key3]-d)/ngram_2_g_aug[key3[0:2]] +lamda_2_g[key3[0:2]]*P_KN_2_g[key3[1:3]]
    else:
        key2 = key3[1:3]
        if (key2 in ngram_2_g_aug) & (key3[0:2] in lamda_2_g):
            prob = lamda_2_g[key3[0:2]]*P_KN_2_g[key3[1:3]]
        else:
            prob = lamda_1_g[key2[0]]*P_CN_1_g[key2[1]]
    return prob

P_KN_3 ={}
for key3 in keys_ngram_3_g_aug:
    P_KN_3[key3] = P_KN_3_g(key3)

# In[]

logP_3_g = 0;
comb_total = 0;
for sents in test_sents_g_aug:
    length = len(sents)
    for i in range(0,length-2):
        comb_total = comb_total+1
        key = (sents[i],sents[i+1],sents[i+2])
        prob = P_KN_3_g(key)
        logP_3_g = logP_3_g + np.log(prob)

ppxty_3_g =np.exp(-logP_3_g/comb_total)
print("trigram perplexity = "+ str(np.floor(ppxty_3_g)))

# In[]

# generating trigram sentence

# in one keyword it will take bigram probability
start_ws = ["extra"]
ws = input("give a key word:--")
start_ws=[ws]
key_bi =[]
val_bi =[]
for key in P_KN_2_g:
    if key[0]==start_ws[0]:
        key_bi.append(key[1])
        val_bi.append(P_KN_2_g[key])
if not len(val_bi) ==0:
    amx_val = np.argmax(val_bi)
    start_ws.append(key_bi[amx_val])
else:
    print("word doesn't exit. selecting start word as i")
    start_ws = ["i","am"]

amx_val = np.argmax(val_bi)
start_ws.append(key_bi[amx_val])
        
#start_ws = ["i","was"]
start_ws_old=start_ws;
gen_sent = []
gen_sent.append(start_ws[0])
gen_sent.append(start_ws[1])
keys_temp =list( P_KN_3.keys())
val_temp = []
for key in keys_temp:
    val_temp.append(P_KN_3[key])


print("trigram Generated sentence is:")
num_token =50
count_token = 0
while(True):
    key_sel = []
    val_sel =[]
    for i in range(0,len(keys_temp)):
        if (keys_temp[i][0] == start_ws[0] ) & (keys_temp[i][1] == start_ws[1] ):
            key_sel.append(i)
            val_sel.append(val_temp[i])
    if len(val_sel)==0:
        print("trigram pair for this bigram doesn't exist")
        break;
    ind_max = key_sel[np.argmax(val_sel)]
    pred_word = keys_temp[ind_max][2]
    gen_sent.append(pred_word)
    start_ws = [start_ws[1],pred_word]
    del keys_temp[ind_max]
    del val_temp[ind_max]
    count_token = count_token + 1
    if (pred_word=='E'):
        start_ws = start_ws_old
    if (count_token==num_token) :
        break


filtered_sent = []
for i in range(0,len(gen_sent)):
    if not ((gen_sent[i] =='E') or (gen_sent[i] =='S')):
        filtered_sent.append(gen_sent[i])

        
print(' '.join(filtered_sent))
            

