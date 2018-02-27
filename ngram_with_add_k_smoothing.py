#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:01:23 2018

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
select_train = "b"
# For gutenberg as train data-> select select_train = "g"
# For brown as train data-> select select_train = "b"
select_test = "b"


# In[]

sentnce_g = gutenberg.sents()
sentnce_b = brown.sents()

# sentence normalization 1 with start S and end E tag augmentation

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

#  uni-gram freq count
for word in train_words_g:
    key = word
    if key in ngram_1_g:
        ngram_1_g[key]=ngram_1_g[key] +1 
    elif key not in ngram_1_g:
        ngram_1_g[key]=1


# In[]
## UNK tag
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
# sentence augmentation/normalization

train_sents_g_aug = []
for sents in train_sents_g:
    sent_chnge =[ x if x in ngram_1_g_aug else 'UNK' for x in sents] 
    train_sents_g_aug.append(sent_chnge)



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



# In[]

# All existing key in dictionary . Those which doesn't exit and exist keynser smoothing
keys_ngram_1_g_aug =list( ngram_1_g_aug.keys())
keys_ngram_2_g_aug =list( ngram_2_g_aug.keys())
keys_ngram_3_g_aug =list( ngram_3_g_aug.keys())



# In[]

# propbability for unigram model with add-K smoothing
n_key_1_g_aug = len(keys_ngram_1_g_aug) 
p_key_1_g_aug= {}
values_ngram_1_g_aug = list(ngram_1_g_aug.values())
n_total = np.sum(values_ngram_1_g_aug)

for i in range(0,n_key_1_g_aug):
    key = keys_ngram_1_g_aug[i]
    p_key_1_g_aug[key] = ngram_1_g_aug[key]/n_total


# In[]
    # add-K smoothing constant
k = 0.001;

# probability with add-K smoothing
n_key_2_g_aug = len(keys_ngram_2_g_aug) 
p_key_2_g_aug= {}
p_not_key_2_g_aug={} # those which doesn't in dictionary -constant value
for i in range(0,n_key_2_g_aug):
    key = keys_ngram_2_g_aug[i]
    p_key_2_g_aug[key] =  (ngram_2_g_aug[key]+k)/(ngram_1_g_aug[key[0]]+k*n_key_1_g_aug)

for i in range(0,n_key_1_g_aug):
    key = keys_ngram_1_g_aug[i]
    p_not_key_2_g_aug[key]= k/(ngram_1_g_aug[key]+k*n_key_1_g_aug)


# In[]
# add-K smoothing constant
k = 0.001;
# probability ngram-3 with add-K smoothing
n_key_3_g_aug = len(keys_ngram_3_g_aug) 
p_key_3_g_aug= {}
p_not_key_3_g_aug={} # those which doesn't in dictionary -constant value
for i in range(0,n_key_3_g_aug):
    key = keys_ngram_3_g_aug[i]
    p_key_3_g_aug[key] =  (ngram_3_g_aug[key]+k)/(ngram_2_g_aug[key[0:2]]+k*n_key_1_g_aug**2)

for i in range(0,n_key_2_g_aug):
    key = keys_ngram_2_g_aug[i]
    p_not_key_3_g_aug[key]= k/(ngram_2_g_aug[key]+k*n_key_1_g_aug**2)



# In[]

# test data sentence augmentation/normalization

test_sents_g_aug = []
for sents in test_sents_g:
    sent_chnge =[ x if x in ngram_1_g_aug else 'UNK' for x in sents] 
    test_sents_g_aug.append(sent_chnge)


# In[]
# unigram perplexity on test data
logP_1_g = 0;
comb_total = 0;
for sents in test_sents_g_aug:
    length = len(sents)
    for i in range(0,length):
        comb_total = comb_total+1
        key = sents[i]
        if key in ngram_1_g_aug:
            logP_1_g = logP_1_g + np.log(p_key_1_g_aug[key])

ppxty_1_g =np.exp(-logP_1_g/comb_total)
print("unigram perplexity = "+ str(np.floor(ppxty_1_g)))
# In[]

# bigram perplexity on test data
logP_2_g = 0;
comb_total = 0;
for sents in test_sents_g_aug:
    length = len(sents)
    for i in range(0,length-1):
        comb_total = comb_total+1
        key = (sents[i],sents[i+1])
        if key in ngram_2_g_aug:
            logP_2_g = logP_2_g + np.log(p_key_2_g_aug[key])
        else:
            key_chk = sents[i]
            logP_2_g = logP_2_g + np.log(p_not_key_2_g_aug[key_chk])

ppxty_2_g =np.exp(-logP_2_g/comb_total)
print("bigram perplexity = "+ str(np.floor(ppxty_2_g)))

# In[]

# trigram perplexicity on test data
logP_3_g = 0;
comb_total = 0;
for sents in test_sents_g:
    length = len(sents)
    for i in range(0,length-2):
        comb_total = comb_total+1
        key = (sents[i],sents[i+1],sents[i+2])
        key2 =(sents[i],sents[i+1]) 
        if key in ngram_3_g_aug:
            logP_3_g = logP_3_g + np.log(p_key_3_g_aug[key])
        elif key2 in p_not_key_3_g_aug:
            logP_3_g = logP_3_g + np.log(p_not_key_3_g_aug[key2])
        else:
            logP_3_g = logP_3_g + np.log(1/(n_key_1_g_aug**2))

ppxty_3_g =np.exp(-logP_3_g/comb_total)
print("trigram perplexity = "+ str(np.floor(ppxty_3_g)))

# In[]

# generating trigram sentence

# in one keyword it will take bigram probability
start_ws = ["i"]
ws = input("give a key word:--")
start_ws=[ws]
key_bi =[]
val_bi =[]
for key in p_key_2_g_aug:
    if key[0]==start_ws[0]:
        key_bi.append(key[1])
        val_bi.append(p_key_2_g_aug[key])
if not len(val_bi) ==0:
    amx_val = np.argmax(val_bi)
    start_ws.append(key_bi[amx_val])
else:
    print("word doesn't exit. selecting start word as i")
    start_ws = ["i","am"]
        
#start_ws = ["i","was"]
start_ws_old=start_ws;
gen_sent = []
gen_sent.append(start_ws[0])
gen_sent.append(start_ws[1])
keys_temp =list( p_key_3_g_aug.keys())
val_temp = []
for key in keys_temp:
    val_temp.append(p_key_3_g_aug[key])


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
            



