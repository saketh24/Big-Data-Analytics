#!/usr/bin/env python
# coding: utf-8

# In[63]:


from pyspark import SparkContext,SparkConf
import numpy as np
from scipy import stats
import sys
import json
import re


# In[64]:


conf = SparkConf()
sc = SparkContext(conf=conf)


# In[65]:


def map_1(data):
    data = json.loads(data)
    text = data["reviewText"]
    text = text.lower()
    words = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', text)
    return words


# In[67]:


rdd = sc.textFile(sys.argv[1]).filter(lambda e: "reviewText" in e).flatMap(map_1).map(lambda word: (word,1)).reduceByKey(lambda a,b:a+b).collect()


# In[68]:


rdd.sort(key = lambda x: x[1])


# In[69]:


top_1k = rdd[-1000:]
top_1k.reverse()


# In[70]:


def find_rel_freq(data,words):
    data = json.loads(data)
    text = data["reviewText"]
    rating = float(data["overall"])
    verified = 1 if str(data['verified']) == "True" else 0
    text = text.lower()
    w_list = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', text)
    final_list = []
    if len(w_list) == 0:
        return []
    for word in words:
        rel_freq = w_list.count(word[0]) / len(w_list)
        final_list.append((word[0], (rel_freq,verified,rating)))
    return final_list


# In[71]:


rdd1 = sc.textFile(sys.argv[1]).filter(lambda e: "reviewText" in e and "verified" in e).flatMap(lambda review: find_rel_freq(review, top_1k)).groupByKey().map(lambda x : (x[0], list(x[1])))


# In[72]:


def lin_reg(data):
    x = []
    y = []
    for vals in data:
        x.append(float(vals[0]))
        y.append(float(vals[2]))
    X = np.reshape(x,(len(x),1))
    Y = np.reshape(y,(len(y),1))
    if len(x)>1:
        X = (X-np.mean(X,axis=0))/ (np.std(X,axis=0))
        Y = (Y-np.mean(Y))/ (np.std(Y))
    X = np.hstack((X,np.ones((len(x),1))))
    X_inverse = np.linalg.pinv(np.dot(np.transpose(X),X))
    w = np.dot(X_inverse,np.transpose(X))
    beta_vals = np.dot(w,Y)
    dof = len(x) - 3
    if dof <= 0:
        return (beta_vals[0,0], -1, -1)
    rss = np.sum(np.power((Y - np.dot(X, beta_vals)), 2))
    s_squared = rss / dof
    temp = (beta_vals[0, 0] / np.sqrt(s_squared / np.sum(np.power((X[:, 0]), 2))))
    p_value = stats.t.sf(np.abs(temp), dof) * 2
    corrected_p_value = p_value * 1000
    return (beta_vals[0,0], corrected_p_value)
    


# In[73]:


def controlled_regression(data):
    x = []
    y = []
    for vals in data:
        x.append([float(vals[0]), float(vals[1])])
        y.append(float(vals[2]))
    X = np.reshape(x,(len(x),2))
    Y = np.reshape(y, (len(y),1))
    if len(x)>1:
        X = (X-np.mean(X,axis=0))/ (np.std(X,axis=0))
        Y = (Y-np.mean(Y))/ (np.std(Y))
    X = np.hstack((X,np.ones((len(x),1))))
    X_inverse = np.linalg.pinv(np.dot(np.transpose(X),X))
    w = np.dot(X_inverse,np.transpose(X))
    beta_vals = np.dot(w,Y)
    dof = len(x) - 3
    if dof <= 0:
        return (beta_vals[0,0], -1, -1)
    rss = np.sum(np.power((Y - np.dot(X, beta_vals)), 2))
    s_squared = rss / dof
    temp = (beta_vals[0, 0] / np.sqrt(s_squared / np.sum(np.power((X[:, 0]), 2))))
    p_value = stats.t.sf(np.abs(temp), dof) * 2
    corrected_p_value = p_value * 1000
    return (beta_vals[0,0], corrected_p_value)


# In[74]:


positive = rdd1.mapValues(lin_reg).takeOrdered(20, lambda res: -res[1][0])
negative = rdd1.mapValues(lin_reg).takeOrdered(20, lambda res: res[1][0])


# In[75]:


positive_controlled = rdd1.mapValues(controlled_regression).takeOrdered(20,lambda x: -x[1][0])
negative_controlled = rdd1.mapValues(controlled_regression).takeOrdered(20,lambda x: x[1][0])


# In[76]:


print("Top 20 positive words: " + str(positive))
print("Top 20 negative words: " + str(negative))
print("Top 20 positive words controlling for verified: " + str(positive_controlled))
print("Top 20 negative words controlling for verified: " + str(negative_controlled))


# In[ ]:




