#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


#CURRENT WORKING DIRECTORY
start_dir = os.getcwd()


# In[3]:


start_dir


# In[4]:


q_num = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20']

# 'fort and temples in historical empires','ramagiri fort and forest'
queries = ['bird population in national park','hindu temples in coastal region','historical village in kerala',
           'geography of lakes in india','new science in modern physics','scientific proofs of modern physics',
           'film of three friends','historical cities of india','german women monarch','set theory indian mathematician',
           'indian literature fiction story','fiction and ghost imaginary comedy novel','inflammatory diseases and their symptoms',
           'age of heart stroke of covid patients','antibiotics to cure dangerous disease','indian COVID-19 pandemic','efficient data structures',
           'some hashing algorithms','role of artificial intelligence',"name some functional programming languages"]


# In[5]:


with open('query.txt','w') as f_q:
    for ind_q in range(20):
        line = q_num[ind_q]+ '\t'+ queries[ind_q] +'\n'
        f_q.write(line)


# In[ ]:




