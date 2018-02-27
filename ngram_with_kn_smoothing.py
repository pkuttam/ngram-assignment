#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:51:47 2018

@author: pk.uttam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:52:55 2018

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
sentnce_g = gutenberg.sents()
sentnce_b = brown.sents()

# gutenberg
sents_g = [];

for sents in sentnce_g:
    sent_lower = [x.lower() for x in sents ] 
    sents_g.append(['S','S']+sent_lower+['E'])

#vocab_g = list(set(word_prep_g))
#brown
sents_b = [];

for sents in sentnce_b:
    sent_lower = [x.lower() for x in sents ] 
    sents_b.append(['S','S']+sent_lower+['E'])

#vocab_b = list(set(word_prep_b))
num_sents_g = len(sents_g)
num_sents_b = len(sents_b)

num_train_g = int(num_sents_g*0.8)
num_test_g = num_sents_g-num_train_g

num_train_b = int(num_sents_b*0.8)
num_test_b = num_sents_b-num_train_b

train_sents_g = sents_g[0:num_train_g]
test_sents_g = sents_g[num_train_g:num_sents_g]

train_sents_b = sents_b[0:num_train_b]
test_sents_b = sents_b[num_train_b:num_sents_b]
del sents_g
del sents_b
del brown
del gutenberg
del sentnce_b
del sentnce_g

train_words_g = list(itertools.chain.from_iterable(train_sents_g))
train_words_b = list(itertools.chain.from_iterable(train_sents_b))
# In[]

ngram_1_g = {}
ngram_1_b = {}
ngram_1_gb = {}

# gutenberg uni-gram freq count
for word in train_words_g:
    key = word
    if key in ngram_1_g:
        ngram_1_g[key]=ngram_1_g[key] +1 
    elif key not in ngram_1_g:
        ngram_1_g[key]=1
    if key in ngram_1_gb:
        ngram_1_gb[key] = ngram_1_gb[key] + 1
    elif key not in ngram_1_gb:
        ngram_1_gb[key] = 1


# brown uni-gram freq count
for word in train_words_b:
    key = word
    if key in ngram_1_b:
        ngram_1_b[key]=ngram_1_b[key] +1 
    elif key not in ngram_1_b:
        ngram_1_b[key]=1
    if key in ngram_1_gb:
        ngram_1_gb[key] = ngram_1_gb[key] + 1
    elif key not in ngram_1_gb:
        ngram_1_gb[key] = 1


# In[]
## UNK tag
max_freq = 10

key_ngram_1_g = list(ngram_1_g.keys())
key_ngram_1_b = list(ngram_1_b.keys())
key_ngram_1_gb = list(ngram_1_gb.keys())

ngram_1_g_aug = dict(ngram_1_g)
ngram_1_b_aug = dict(ngram_1_b)
ngram_1_gb_aug = dict(ngram_1_gb)

ngram_1_g_aug['UNK'] = 0
ngram_1_b_aug['UNK'] = 0
ngram_1_gb_aug['UNK'] = 0

for word in key_ngram_1_g:
    if ngram_1_g[word]<=max_freq:
        #print(ngram_1_g[word])
        ngram_1_g_aug['UNK']=ngram_1_g_aug['UNK']+ngram_1_g[word]
        del ngram_1_g_aug[word]

for word in key_ngram_1_b:
    if ngram_1_b[word]<=max_freq:
        ngram_1_b_aug['UNK']=ngram_1_b_aug['UNK']+ngram_1_b[word]
        del ngram_1_b_aug[word]

for word in key_ngram_1_gb:
    if ngram_1_gb[word]<=max_freq:
        ngram_1_gb_aug['UNK']=ngram_1_gb_aug['UNK']+ngram_1_gb[word]
        del ngram_1_gb_aug[word]



# In[]
train_sents_gb = train_sents_g + train_sents_b
# gutenberg sentence augmentation/normalization

train_sents_g_aug = []
for sents in train_sents_g:
    sent_chnge =[ x if x in ngram_1_g_aug else 'UNK' for x in sents] 
    train_sents_g_aug.append(sent_chnge)

#brown
train_sents_b_aug = []
for sents in train_sents_b:
    sent_chnge =[ x if x in ngram_1_b_aug else 'UNK' for x in sents] 
    train_sents_b_aug.append(sent_chnge)

# gtenberg+brown
train_sents_gb_aug = []
for sents in train_sents_gb:
    sent_chnge =[ x if x in ngram_1_gb_aug else 'UNK' for x in sents] 
    train_sents_gb_aug.append(sent_chnge)


# In[]
# Bigram count for gutenberg
ngram_2_g_aug={}
for sents in train_sents_g_aug:
    length = len(sents)
    for i in range(0,length-1):
        key = (sents[i],sents[i+1])
        if key in ngram_2_g_aug:
            ngram_2_g_aug[key] = ngram_2_g_aug[key]+1
        else:
            ngram_2_g_aug[key] = 1

# Bigram count for brown
ngram_2_b_aug={}
for sents in train_sents_b_aug:
    length = len(sents)
    for i in range(0,length-1):
        key = (sents[i],sents[i+1])
        if key in ngram_2_b_aug:
            ngram_2_b_aug[key] = ngram_2_b_aug[key]+1
        else:
            ngram_2_b_aug[key] = 1

# Bigram count for gutenberg +brown
ngram_2_gb_aug={}
for sents in train_sents_gb_aug:
    length = len(sents)
    for i in range(0,length-1):
        key = (sents[i],sents[i+1])
        if key in ngram_2_gb_aug:
            ngram_2_gb_aug[key] = ngram_2_gb_aug[key]+1
        else:
            ngram_2_gb_aug[key] = 1

# In[]
# trigram count for gutenberg
ngram_3_g_aug={}
for sents in train_sents_g_aug:
    length = len(sents)
    for i in range(0,length-2):
        key = (sents[i],sents[i+1],sents[i+2])
        if key in ngram_3_g_aug:
            ngram_3_g_aug[key] = ngram_3_g_aug[key]+1
        else:
            ngram_3_g_aug[key] = 1

# trigram count for brown
ngram_3_b_aug={}
for sents in train_sents_b_aug:
    length = len(sents)
    for i in range(0,length-2):
        key = (sents[i],sents[i+1],sents[i+2])
        if key in ngram_3_b_aug:
            ngram_3_b_aug[key] = ngram_3_b_aug[key]+1
        else:
            ngram_3_b_aug[key] = 1

# trigram count for gutenberg+brown
ngram_3_gb_aug={}
for sents in train_sents_gb_aug:
    length = len(sents)
    for i in range(0,length-2):
        key = (sents[i],sents[i+1],sents[i+2])
        if key in ngram_3_gb_aug:
            ngram_3_gb_aug[key] = ngram_3_gb_aug[key]+1
        else:
            ngram_3_gb_aug[key] = 1



# In[]

# All existing key in dictionary . Those which doesn't exit and exist keynser smoothing
keys_ngram_1_g_aug =list( ngram_1_g_aug.keys())
keys_ngram_1_b_aug =list( ngram_1_b_aug.keys())
keys_ngram_1_gb_aug =list( ngram_1_gb_aug.keys())

keys_ngram_2_g_aug =list( ngram_2_g_aug.keys())
keys_ngram_2_b_aug =list( ngram_2_b_aug.keys())
keys_ngram_2_gb_aug =list( ngram_2_gb_aug.keys())

keys_ngram_3_g_aug =list( ngram_3_g_aug.keys())
keys_ngram_3_b_aug =list( ngram_3_b_aug.keys())
keys_ngram_3_gb_aug =list( ngram_3_gb_aug.keys())




# In[]

# test data gutenberg sentence augmentation/normalization

test_sents_g_aug = []
for sents in test_sents_g:
    sent_chnge =[ x if x in ngram_1_g_aug else 'UNK' for x in sents] 
    test_sents_g_aug.append(sent_chnge)

#brown
test_sents_b_aug = []
for sents in test_sents_b:
    sent_chnge =[ x if x in ngram_1_b_aug else 'UNK' for x in sents] 
    test_sents_b_aug.append(sent_chnge)

# In[]

#KN for unigram
d = 0.75

# unigram continuation
P_CN_1_g_nemu = {}
P_CN_1_g_deno = {}
P_CN_1_g = {}

for key2 in keys_ngram_2_g_aug:
    if key2[1] in P_CN_1_g_nemu:
        P_CN_1_g_nemu[key2[1]] = P_CN_1_g_nemu[key2[1]] + 1
        P_CN_1_g_deno[key2[1]] = P_CN_1_g_deno[key2[1]] + ngram_1_g_aug[key2[0]]
    else:
        P_CN_1_g_nemu[key2[1]] = 1
        P_CN_1_g_deno[key2[1]] = ngram_1_g_aug[key2[0]]

for key in P_CN_1_g_nemu:
    P_CN_1_g[key] =   P_CN_1_g_nemu[key]/P_CN_1_g_deno[key]
      


# unigram lamda
lamda_1_g_nemu = {}
for key2 in keys_ngram_2_g_aug:
    if key2[0] in lamda_1_g_nemu:
        lamda_1_g_nemu[key2[0]] = lamda_1_g_nemu[key2[0]] + 1
    else:
        lamda_1_g_nemu[key2[0]] =  1

lamda_1_g={}
for key in lamda_1_g_nemu:
    lamda_1_g[key] = d*lamda_1_g_nemu[key]/ngram_1_g_aug[key]

# unigram continuation for brown
P_CN_1_b_nemu = {}
P_CN_1_b_deno = {}
P_CN_1_b = {}

for key2 in keys_ngram_2_b_aug:
    if key2[1] in P_CN_1_b_nemu:
        P_CN_1_b_nemu[key2[1]] = P_CN_1_b_nemu[key2[1]] + 1
        P_CN_1_b_deno[key2[1]] = P_CN_1_b_deno[key2[1]] + ngram_1_b_aug[key2[0]]
    else:
        P_CN_1_b_nemu[key2[1]] = 1
        P_CN_1_b_deno[key2[1]] = ngram_1_b_aug[key2[0]]

for key in P_CN_1_b_nemu:
    P_CN_1_b[key] =   P_CN_1_b_nemu[key]/P_CN_1_b_deno[key]
      


# unigram lamda
lamda_1_b_nemu = {}
for key2 in keys_ngram_2_b_aug:
    if key2[0] in lamda_1_b_nemu:
        lamda_1_b_nemu[key2[0]] = lamda_1_b_nemu[key2[0]] + 1
    else:
        lamda_1_b_nemu[key2[0]] =  1

lamda_1_b={}
for key in lamda_1_b_nemu:
    lamda_1_b[key] = d*lamda_1_b_nemu[key]/ngram_1_b_aug[key]


# probability KN
P_KN_2_g = {}
for key2 in keys_ngram_2_g_aug:
    P_KN_2_g[key2] = (ngram_2_g_aug[key2] -d)/ngram_1_g_aug[key2[0]] + lamda_1_g[key2[0]]*P_CN_1_g[key2[1]]


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
print(ppxty_2_g)

logP_2_b = 0;
comb_total = 0;
for sents in test_sents_b_aug:
    length = len(sents)
    for i in range(0,length-1):
        comb_total = comb_total+1
        key = (sents[i],sents[i+1])
        if key in ngram_2_b_aug:
            prob = (ngram_2_b_aug[key] -d)/ngram_1_b_aug[key[0]] + lamda_1_b[key[0]]*P_CN_1_b[key[1]]
            logP_2_b = logP_2_b + np.log(prob)
        else:
            prob = lamda_1_b[key[0]]*P_CN_1_b[key[1]]
            logP_2_b = logP_2_b + np.log(prob)

ppxty_2_b =np.exp(-logP_2_b/comb_total)
print(ppxty_2_b)

# In[]
