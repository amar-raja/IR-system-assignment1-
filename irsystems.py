#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from collections import Counter
import math
import pickle
import sys


# In[2]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


#for google colab
# os.chdir('/content')
# pwd = os.getcwd()
# ds = r"/drive/MyDrive/Colab Notebooks/english-corpora/english-corpora"
# os.chdir(pwd +ds)


# In[ ]:





# In[4]:


#Reaching Corpus Directory
pwd = os.getcwd()

start_dir = pwd

ds = r"\english-corpora\english-corpora"
os.chdir(pwd +ds)


# In[5]:


#Names of all Docs
files = os.listdir()

#LENGTH OF CORPUS
Corpus = len(files)                    


# In[6]:


#=REMOVING NONASCCI CHAR
def rem_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


# In[7]:


#PUNCTUATION SYMBOLS
punc = string.punctuation                      
def remove_punc(text):
    return text.translate(str.maketrans('','',punc))

#STOPWORDS
s_words = stopwords.words("english")

#STEMMER
stemmer = PorterStemmer()


# In[8]:


def remove_stopwords_and_stemming(text):                        #and stemming
    tokens = word_tokenize(text)                                     #TOKENIZATION
#     print(tokens)
    newtext = []
    for w in tokens:
        if w not in s_words:
            w = stemmer.stem(w)                     #STEMMING
            newtext.append(w)
    return newtext
#     return " ".join(newtext)
            


# In[9]:


#TERM FREQUENCY  (NUMBER OF OCCURENCES ONLY)
tf=[]                            #LIST of DICTINARIES which contain termfreq for each "document".
def term_freq(text):             #text should be in list
#     length = len(text)          
    tf.append(Counter(text))

    


# In[10]:


df_posting_list={}
doc_num = 0

#DOCUMENT FREQUENCY
def doc_freq(text):
    global doc_num
    for i in range(len(text)):
        try:
            df_posting_list[text[i]].add(doc_num)
        except:
            df_posting_list[text[i]] = {doc_num}
    doc_num+=1


# ## TF-IDF

# In[11]:


#TF-IDF calculation docwise
# doc=0
tf_idf_mat = {}
l2_norms=[]

def tf_idf():
    
    for doc in range(Corpus):
        sum=0
        for word in tf[doc]:
            tf_normalised = tf[doc][word] / len(tf[doc])
    #         print(tf_normalised)
    #         print(df[word])
            idf = math.log((Corpus/df[word]+1),2)

            value = tf_normalised*idf
            
            sum+= value**2
            tf_idf_mat[doc,word] = value

        sum = math.sqrt(sum)
        l2_norms.append(sum)


# In[12]:


#LENGTH OF DOCUMENTS
length_of_doc = []


# ## Preprocessing Corpus and calculating "Term Freq" and "Doc Freq"

# In[ ]:



for document in files:
    with open(document,'r',encoding="utf8") as f:
        t = ""
#         length = 0
        for i in f:                                       #LINE BY LINE
            l = i.strip().lower()                        #REMOVING TRAILING SPACES(and NEWLINE "\n") AND LOWERCASE LETTERS
            l = rem_nonascii(l)                          #REMOVING NON-ASCII CHAR
            t += "".join(l)
            t+=" "
#             length += 1
        
        t = t.strip()                                    # #REMOVING TRAILING SPACE
        t = remove_punc(t)
        t = remove_stopwords_and_stemming(t)
        length = len(t)
        #TERM FREQUENCY
        term_freq(t)

        #CALLING THE DOCUMENT FREQUENCY FUNCTION
        doc_freq(t)        
        
        #length
        length_of_doc.append(length)


# In[ ]:


# COUNTING DOCUMENT COUNT for each term in the vocabulary of corpus
df = df_posting_list.copy()
for i in df:
    df[i] = len(df[i])


# In[ ]:





# In[ ]:


#CALLING TF_IDF
tf_idf()                     #must be called for creating above df only, as df is used in tf-idf


# In[ ]:


total_vocab = [w for w in df]             #TOTAL UNIQUE TOKENS IN THE CORPUS


# In[ ]:


#CHANGING DIRECTORY
os.chdir(start_dir)


# In[ ]:





# In[ ]:


#40 query filename must be placed here
query_file_name = sys.argv[1]  


# In[ ]:


print(query_file_name)


# In[ ]:


#checking query
# query_file_name = 'query_checking.txt'


# In[ ]:


#RANKING LISTS OF EACH QUERY
ranking_tfidf = []
ranking_bm25 = []
ranking_boolean = []


# In[ ]:


num=1
with open(query_file_name,'r',encoding="utf8") as file_query:
    for i in file_query:
        
        qnum , query = i.split(maxsplit=1)
        
        query = query.strip().lower()
        query = remove_punc(query)
        query = remove_stopwords_and_stemming(query)
        
        #tf for query
        Q = {}
        tokens = Counter(query)
        l2_norm_q = 0
        for w,f in tokens.items():
            tf_q = f / len(tokens) 
            try:
                df_q = df[w]
            except:
                df_q = 0
            idf_q = math.log(((Corpus)/(df_q+1)),2)

            tf_idf_q = tf_q * idf_q
            l2_norm_q += tf_idf_q**2
            Q[w] = tf_idf_q

        l2_norm_q = math.sqrt(l2_norm_q)
        
        
        
        similarity_scores=[]
        for document in range(Corpus):
            numerator = 0
            cos_sim=0
            for w,f in Q.items():
                try:
                    numerator += tf_idf_mat[document,w]*f
                except:
                    pass
            cos_sim = numerator / (l2_norms[document]*l2_norm_q)
            similarity_scores.append(cos_sim)

        r_arr = np.array(similarity_scores)
        ind = np.argsort(r_arr)[::-1][:10]

        j=0
#         print("for QUERY ",qnum)
        
        rank_list=[]
        for i in ind:
            rank_list.append(files[i][:-4])
#             print("rank {} is {},".format(j+1,files[i]))
            j+=1
#         print()
        ranking_tfidf.append(rank_list)
        


# ## BM25

# In[ ]:


L = sum(length_of_doc)/Corpus
#Tuned parameters
k=2
b=0.75

with open(query_file_name,'r',encoding="utf8") as file_query:
    for i in file_query:
        
        qnum , query = i.split(maxsplit=1)
        
        query = query.strip().lower()
        query = remove_punc(query)
        query = remove_stopwords_and_stemming(query)
        
        tokens = Counter(query)
        
        similarity_scores_bm25 = []

        for document in range(Corpus):
            score=0
            for w,f in tokens.items():
                idf = math.log((Corpus-df[w]+0.5)/(df[w]+0.5))
                num = (k+1)*tf[document][w]
                denom = tf[document][w] + (k*(1-b +(b * length_of_doc[document]/L)))
                score += (idf * num)/denom
                score = score*f

            similarity_scores_bm25.append(score)
        
        r_arr_bm25 = np.array(similarity_scores_bm25)
        ind_bm25 = np.argsort(r_arr_bm25)[::-1][:10]
        
        j=0
        rank_list=[]
        
#         print("for QUERY ",qnum)
        for i in ind_bm25:
            rank_list.append(files[i][:-4])
#             print("rank {} is {},".format(j+1,files[i]))
            j+=1
#         print()
        ranking_bm25.append(rank_list)


# In[ ]:


# sorted(range(len(r_arr_bm25)),key=r_arr_bm25.__getitem__)[::-1][:10]

# ind_bm25


# ## BOOLEAN RETRIEVAL

# In[ ]:


with open(query_file_name,'r',encoding="utf8") as file_query:
    c=0
    for i in file_query:
        
        qnum , query = i.split(maxsplit=1)
        
        query = "javascript"
        
        query = query.strip().lower()
        query = remove_punc(query)
        query = remove_stopwords_and_stemming(query)
        
        #Storing  Frequency of Docs
        relevancy = [0] * Corpus
        for w in query:
            posting_list = df_posting_list[w]
            for i in posting_list:
                relevancy[i] += 1
        
#         r_arr_bool = np.array(relevancy)
#         r=r_arr_bool.copy()
        # ind_bool = np.argsort(r_arr_bool,kind='stable')
#         ind_bool = np.argsort(r_arr_bool)[::-1][:10]

        #correct
        rel_pair = []
        for i in range(len(relevancy)):
            rel_pair.append([i,relevancy[i]])
        
        rel_pair.sort(key = lambda x:x[1],reverse=True)
        ind_bool = rel_pair[:10]
        
        rank_list=[]
#         print("for QUERY ",qnum)
        for i in ind_bool:
            rank_list.append(files[i[0]][:-4])
#             print("rank {} is {},".format(j+1,files[i]))

#         print()
        ranking_boolean.append(rank_list)


# In[ ]:





# In[ ]:


print(ranking_tfidf[0],end="")
print()

print(ranking_bm25[0],end="")


# In[ ]:


print(ranking_boolean[0],end="")


# In[ ]:





# ## Returning top-10 docs for each system for given 40 queries

# In[ ]:


q_num_40 = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10',
        'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20',
        'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29','Q30',
        'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39','Q40']

#output filenames
docnames_rank_40 = ['boolean40.txt','tfidf40.txt','bm2540.txt']       


ranking_three_systems = [ranking_boolean,ranking_tfidf,ranking_bm25]


# In[ ]:


for name in range(len(docnames_rank_40)):
    with open(docnames_rank_40[name],'w') as f_qrel:
        for num in range(len(q_num_40)):
            for q in range(10):
                line = q_num_40[num] + ',1,' + ranking_three_systems[name][num][q] +',1' +'\n'
                f_qrel.write(str(line))


# In[ ]:





# In[ ]:




