#!/usr/bin/env python
# coding: utf-8

# In[338]:


from pyspark import SparkContext,SparkConf
import csv
import numpy as np
from scipy import stats
import sys
import json
import re


# In[337]:


conf = SparkConf()
sc = SparkContext(conf=conf)


# In[339]:


def map_1(data):
    data = json.loads(data)
    return ((data["asin"],data["reviewerID"]),float(data["overall"]))


# In[340]:


def reduce_vals(data):
    l = list(data[1])
    return((data[0][0],(data[0][1],float(l[-1]))))


# In[341]:


def flatten(data):
    final_list = []
    for vals in data[1]:
        final_list.append((vals[0],(data[0],vals[1])))
    return final_list


# In[342]:


rdd = sc.textFile(sys.argv[1]).map(map_1).groupByKey().map(reduce_vals)


# In[343]:


filter1 = rdd.groupByKey().mapValues(lambda x: list(x)).filter(lambda x: len(x[1]) >= 25).flatMap(flatten).groupByKey().mapValues(lambda x: list(x)).filter(lambda x: len(x[1]) >= 5).flatMap(flatten).groupByKey().mapValues(lambda x: list(x))


# In[344]:


userids = filter1.flatMap(flatten).map(lambda y: (y[0])).collect()


# In[345]:


unique_users = list(set(userids))
user_list = sc.broadcast(unique_users)


# In[ ]:


i_codes = sys.argv[2]
i_codes = eval(i_codes)
i1 = i_codes[0]
i2 = i_codes[1]


# In[346]:


q1 = filter1.filter(lambda t: t[0] == i1).values().collect()
q2 = filter1.filter(lambda t: t[0] == i2).values().collect()


# In[347]:


item1 = sc.broadcast(q1[0])
item2 = sc.broadcast(q2[0])


# In[348]:


l1 = [float(0) for i in range(len(unique_users))]
l2 = [float(0) for i in range(len(unique_users))]
for user in unique_users:
    if user in dict(q1[0]).keys():
        l1[unique_users.index(user)] = dict(q1[0])[user]
    if user in dict(q2[0]).keys():
        l2[unique_users.index(user)] = dict(q2[0])[user]


# In[349]:


item_vector1 = sc.broadcast(l1)
item_vector2 = sc.broadcast(l2)


# In[350]:


def map_cos(data):
    count = 0
    for values in item1.value:
        if values[0] in dict(data[1]).keys():
            count += 1
        if count >=2:
            break
    if count < 2:
        return (data[0],-1)
    i_vec = [float(0) for i in range(len(user_list.value))]
    for user in user_list.value:
        if user in dict(data[1]).keys():
            i_vec[user_list.value.index(user)] = dict(data[1])[user]
    x1 = np.asarray(item_vector1.value)
    x2 = np.asarray(i_vec)
    cosine_sim = float(1 - (x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))))
    print(cosine_sim)
    if cosine_sim <= 0 :
        return(data[0],-1)
    return(data[0], (cosine_sim, i_vec))


# In[351]:


def map_cos2(data):
    count = 0
    for values in item2.value:
        if values[0] in dict(data[1]).keys():
            count += 1
        if count >=2:
            break
    if count < 2:
        return (data[0],-1)
    i_vec = [float(0) for i in range(len(user_list.value))]
    for user in user_list.value:
        if user in dict(data[1]).keys():
            i_vec[user_list.value.index(user)] = dict(data[1])[user]
    x1 = np.asarray(item_vector2.value)
    x2 = np.asarray(i_vec)
    cosine_sim = float(1 - (x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))))
    print(cosine_sim)
    if cosine_sim <= 0 :
        return(data[0],-1)
    return(data[0], (cosine_sim, i_vec))


# In[352]:


rec1 = filter1.map(map_cos).filter(lambda x: x[1] != -1).collect()
rec2 = filter1.map(map_cos2).filter(lambda x: x[1] != -1).collect()


# In[353]:


to_predict1 = []
predictions1 = []
for user,val in zip(unique_users,l1):
    if val == 0:
        to_predict1.append(user)
for user in to_predict1:
    idx = unique_users.index(user)
    sop = 0
    tot = 0
    user_count = 0
    for rec in rec1:
        if rec[1][1][idx] !=0:
            user_count += 1
            sop += rec[1][1][idx] * rec[1][0]
            tot += rec[1][0]
    if user_count < 2:
        continue
    if sop == 0:
        continue
    predictions1.append((user, round(sop/tot, 2)))
print("Predictons for item " + str(i1) +": \n"  + str(predictions1))


# In[354]:


to_predict2 = []
predictions2 = []
for user,val in zip(unique_users,l2):
    if val == 0:
        to_predict2.append(user)
for user in to_predict2:
    idx = unique_users.index(user)
    sop = 0
    tot = 0
    user_count = 0
    for rec in rec2:
        if rec[1][1][idx] !=0:
            user_count += 1
            sop += rec[1][1][idx] * rec[1][0]
            tot += rec[1][0]
    if user_count < 2:
        continue
    if sop == 0:
        continue
    predictions2.append((user, round(sop/tot, 2)))
print("Predictons for item " + str(i2) + ": \n"  + str(predictions2))


# In[ ]:




