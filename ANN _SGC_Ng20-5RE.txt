#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all the required modules

from importlib import reload
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os
import errno
import string
from nltk.corpus import reuters
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.text import TextCollection
import collections
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from prettytable import PrettyTable
import time
from datetime import timedelta
from sklearn.metrics import recall_score,precision_score,average_precision_score,f1_score,accuracy_score,roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score,completeness_score
import statistics
import math
import sklearn.metrics 
from sklearn.model_selection import KFold
import csv
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from scipy.spatial import distance
import statistics
from scipy.stats import pearsonr,entropy
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedKFold
from scipy.io import arff



from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D
from tensorflow.keras.layers import Dense,BatchNormalization,Flatten,Dropout

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import tensorflow as tf
import string
from sklearn.preprocessing import LabelEncoder
#from keras.optimizers import SGD
from numba import cuda
import gc 
import torch
import pickle as pkl
from time import perf_counter
from tensorflow.keras.optimizers import Adam
import sys
import scipy.sparse as sp
from tensorflow.keras.optimizers import Adam,RMSprop
from numpy.random import seed
from tensorflow.python.keras import backend as K
#seed(0)
#tf.random.set_seed(0)
def sparse_to_torch_sparse(sparse_mx, device='cuda'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    if device == 'cuda':
        indices = indices.cuda()
        values = torch.from_numpy(sparse_mx.data).cuda()
        shape = torch.Size(sparse_mx.shape)
        adj = torch.cuda.sparse.FloatTensor(indices, values, shape)
    elif device == 'cpu':
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
    return adj

def sparse_to_torch_dense(sparse, device='cuda'):
    dense = sparse.todense().astype(np.float32)
    torch_dense = torch.from_numpy(dense).to(device=device)
    return torch_dense

def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    index_dict = {}
    label_dict = {}
    phases = ["train", "val", "test"]
    objects = []
    def load_pkl(path):
        with open(path.format(dataset_str, p), 'rb') as f:
            if sys.version_info > (3, 0):
                return pkl.load(f, encoding='latin1')
            else:
                return pkl.load(f)

    for p in phases:
        index_dict[p] = load_pkl("data/ind.{}.{}.x".format(dataset_str, p))
        label_dict[p] = load_pkl("data/ind.{}.{}.y".format(dataset_str, p))
        print(index_dict[p])
        print(label_dict[p])

    adj = load_pkl("data/ind.{}.BCD.adj".format(dataset_str))
    adj = adj.astype(np.float32)
    adj = preprocess_adj(adj)

    return adj, index_dict, label_dict

def loadData(dataset_str):

    """
    Load Citation Networks Datasets.
    """
    sp_adj, index_dict, label_dict = load_corpus(dataset_str)
    for k, v in label_dict.items():
        if dataset_str == "mr":
            label_dict[k] = torch.Tensor(v)
        else:
            label_dict[k] = torch.LongTensor(v)
    features = list(range(0,sp_adj.shape[0]))

    adj = sparse_to_torch_sparse(sp_adj, device='cpu')
    adj_dense = sparse_to_torch_dense(sp_adj, device='cpu')
    degree=1
    feat_dict, precompute_time = sgc_precompute(adj, adj_dense, degree, index_dict)
    print("pre compute time",precompute_time)
    return feat_dict, label_dict, precompute_time
    
def sgc_precompute(adj, features, degree, index_dict):
    assert degree==1,"Only supporting degree 2 now"
    feat_dict = {}
    start = perf_counter()
    train_feats = features[:, index_dict["train"]]
    train_feats = torch.spmm(adj, train_feats).t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
    feat_dict["train"] = train_feats
    for phase in ["test", "val"]:
        feats = features[:, index_dict[phase]]
        feats = torch.spmm(adj, feats).t()
        feats = feats[:, useful_features_dim]
        feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!
    precompute_time = perf_counter()-start
    return feat_dict, precompute_time


def normalized_adj(adj):
    adj = adj
    adj = sp.coo_matrix(adj)
    print("Sp coo ",adj.shape)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.float_power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return np.transpose(d_mat_inv_sqrt.dot(adj)).dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalized_adj(adj)
    return adj_normalized


def loadR52Dataset(documents,labels):
    categories_list=['acq','alum','bop','carcass','cocoa','coffee','copper','cotton','cpi','cpu','crude','dlr','earn','fuel','gas','gnp','gold',
    'grain','heat','housing','income','instal-debt','interest','ipi','iron-steel','jet','jobs','lead','lei','livestock','lumber',
    'meal-feed','money-fx','money-supply','nat-gas','nickel','orange','pet-chem','platinum','potato','reserves','retail',
    'rubber','ship','strategic-metal','sugar','tea','tin','trade','veg-oil','wpi','zinc'
    ]
    
    docCount=0
    for i in range(0,len(categories_list)):
        category_docs = reuters.fileids(categories_list[i])
        print (categories_list[i])
        for document_id in reuters.fileids(categories_list[i]):
            if(len(reuters.categories(document_id))==1):
                content=str(reuters.raw(document_id))
                soup = BeautifulSoup(content)
                content=soup.get_text()
                documents.append(content)
                docCount+=1
                labels.append(str(reuters.categories(document_id)[0]))
                

def most_common(lst):
    return max(set(lst), key=lst.count)
def tokenize1(documents):
    tokens=[]
    content= documents
    tokens=(word_tokenize(content))
    tokens= [token.lower() for token in tokens ]
    tokens = [token for token in tokens if token not in stopwords]
    tokens= [token for token in tokens if token.isalpha()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens= [token for token in tokens if len(token)>3 ]
    return tokens
def count_values_in_range(a, range_min, range_max):

    # "between" returns a boolean Series equivalent to left <= series <= right.
    # NA values will be treated as False.
    return ((range_min <= a) & (a <= range_max)).sum()
def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
def sorted_tfs(tfs,n):
    doc,terms=tfs.shape
    ind=(np.argsort(-(np.asarray(tfs.sum(axis=0)).ravel())))
    scores=np.zeros((doc,n))
    for i in range(0,len(documents)):
        for j in range(0,n):
            if(tfs[i,ind[j]]!=0):
                scores[i,j]=tfs[i,ind[j]]
    return scores

def dislay_tfidf(vectorizer,tfidf_result):
    print(vectorizer.get_feature_names())
    print(tfidf_result)
def Manhattan(doc1,doc2):
    return distance.cityblock(doc1,doc2)                       
def Euclidean(a, b):#distance
    return distance.euclidean(a,b)
def Minkowski(doc1,doc2):
    return distance.minkowski(doc1,doc2) 
def Cosine(a, b):#distance
    return distance.cosine(a,b)
def Jaccard(a, b):#distance
    return distance.jaccard(a,b)
def EnhancedJaccard(a, b):#distance
    a=[0 if x==0 else 1 for x in a]
    b=[0 if x==0 else 1 for x in b]
    return distance.jaccard(a,b)
def PCC(a, b):
    pcc, col=pearsonr(a,b)
    return abs(pcc)
def extendedJaccard(a,b):
    vector1=[0 if x==0 else 1 for x in a]
    vector2=[0 if x==0 else 1 for x in b]
    dot=np.dot(vector1,vector2)
    sum1=np.sum(vector1)
    sum2=np.sum(vector2)
    denom=math.sqrt(sum1)+math.sqrt(sum2)-dot
    if(denom!=0):
        return 1.0 - (float(dot)/(denom))
    else:
        return -1
def bhatta(a,b):

    length=len(a)
    score = 0;
    score=np.sum(np.sqrt( np.multiply(a,b) ))
    distance=-1*np.log(score)
    return distance;        

def JS(a, b):
   # normalize
    p = a/ np.sum(a)
    q = b/ np.sum(b)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def Pairwise(a,b):
    percentage=1 #100 percent
    K1=np.count_nonzero(a)
    K2=np.count_nonzero(b)
    k=percentage*min(K1,K2)
   # setA=set((-a).argsort()[:k])
   # setB=set((-b).argsort()[:k])
    
    setA=set(np.argpartition(a, len(a)-1 - k)[-k:])
    setB=set(np.argpartition(b, len(b)-1 - k)[-k:])
    union=setA.union(setB)
    elementsA=[a[ind] for ind in union]
    elementsB=[b[ind] for ind in union]
    dist= distance.cosine(elementsA, elementsB)
    return dist

def Dice(a, b):
    a=[(a.astype(bool)).astype(int)]
    b=[(b.astype(bool)).astype(int)]
    return distance.dice(a,b)
def Extended_Dice(a, b):
    a=[0 if x==0 else 1 for x in a]
    b=[0 if x==0 else 1 for x in b]
    sim=(2*np.dot(a,b))/(sum(a)**2 + sum(b)**2)
    return sim
def simIT(a,b,termOcc,docOcc):
    p1= np.divide(a,termOcc) 
    p2= np.divide(b,termOcc)
    minVal=np.minimum(p1,p2)
    pi=np.log((np.array(docOcc)/totalDocs))
    sIT=(2*np.sum(np.multiply(minVal,pi)))/((np.sum(np.multiply(p1,pi)))+(np.sum(np.multiply(p2,pi))))
    return sIT

def ISC(doc1,doc2): 
    dot=sum(np.sqrt(np.multiply(doc1,doc2)))
    isc=dot /( math.sqrt(np.linalg.norm(doc1, ord=1))*math.sqrt(np.linalg.norm(doc2, ord=1)))
    return isc

def SP(doc1,doc2,N):
    
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    nonzeroTerm=list(a.intersection(b))
    minArr=np.minimum(doc1[nonzeroTerm],doc2[nonzeroTerm])
    maxArr=np.maximum(doc1[nonzeroTerm],doc2[nonzeroTerm])
    ln=len(nonzeroTerm)
    docCount=np.zeros(ln)
    aData=allData[:,nonzeroTerm] #extracting data with only feature indices of doc1 intersection  doc2
    for ti in range(0,ln):
        docCount[ti]=count_values_in_range(aData[:,ti], minArr[ti], maxArr[ti]) #count values which lies between min and max
    normFactor= len(a.union(b))
    dCount=np.array(docCount[np.nonzero(docCount)[0]])
    SP_val=0
    if(normFactor!=0):
        SP_val=sum(np.log(totalDocs/dCount))
        SP_val = (SP_val/normFactor)
    return SP_val

def PDSM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=sum(np.minimum(doc1,doc2))
    union=sum(np.maximum(doc1,doc2))
    PF=len(a.intersection(b))
    M=len(doc1)
    AF=len(set(range(0,M))-(a.union(b)))
    psdm=(intersection/union)*((PF+1)/(M-AF+1))
    return psdm
def EPDSM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=np.sum(np.minimum(doc1,doc2))
    union=np.sum(np.maximum(doc1,doc2))
    PF=len(a.intersection(b))
    M=len(doc1)
    AF=M-len(a.union(b))
    psdm=(intersection/union)*((PF+1)/(M-AF+1))
    return psdm
def EEnhancedJaccard(a, b):#similarity
    a=set(np.nonzero(a)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(b)[0])  #indices of non zero elements in doc2
    union=a.union(b)
    intersection=a.intersection(b)
    sim=len(intersection)/len(union)#similarity
    return sim
def smtp(doc1,doc2,var):
    lemda=0.0001
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=a.intersection(b) # indices where both docs have non zero elements
    union=a.union(b) # indices where either doc has non zero elements
    d1=np.array(list(a-intersection)) # doc1 !=0 and doc2=0
    d2=np.array(list(b-intersection)) # doc1 =0 and doc2!=0
    doc1=np.array(doc1)
    doc2=np.array(doc2)
    Nstar=0
    intersection=np.array(list(intersection))
    if (len(intersection)>0):
        term1=np.exp(-1*np.square(( doc1[intersection]-doc2[intersection] )/var[intersection]))
        Nstar=sum(0.5* (1+term1)) +lemda* -1 *(len(d1)+len(d2))
    else:
        Nstar=lemda* -1 *(len(d1)+len(d2))   
    Nunion=len(intersection)+len(d1)+len(d2)
    smtp=0
    if Nunion!=0:
        smtp=((Nstar/Nunion)+lemda)/(1+lemda)
    return smtp
def EISC(doc1,doc2): 
    dot=np.sum(np.sqrt(np.multiply(doc1,doc2)))
    isc=dot /( math.sqrt(np.linalg.norm(doc1, ord=1))*math.sqrt(np.linalg.norm(doc2, ord=1)))
    return isc

def NSMT(doc1,doc2):
    Dij=Nij=Di=Ni=Nj=Dj=0
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=a.intersection(b)# indices where both docs have non zero elements
    a=np.array(list(a-intersection)) 
    b=np.array(list(b-intersection))
    intersection=np.array(list(intersection))
    if len(intersection)>0:
      #  prod=np.multiply(doc1[intersection],doc2[intersection])
        Dij=np.dot(doc1[intersection],doc2[intersection])#sum(prod)
    if len(a)>0:
        Di=sum(doc1[a])
    if len(b)>0:
        Dj=sum(doc2[b])
    Nij=len(intersection)
    Ni=len(a)
    Nj=len(b)
    Nsmt=0
    if (Ni*Di+Dj*Nj)!=0:
        Nsmt=(Nij*Dij)/(Ni*Di+Dj*Nj)
    return Nsmt
def CSMB(doc1,doc2,alpha=0.5,Beta=0.5):
    sim1=CSM_P1(doc1,doc2)
    sim2=CSM_P2(doc1,doc2)
    simValue=alpha*sim1+Beta*sim2
    return simValue
def CSMB_MinMax(doc1,doc2,alpha=0.5,Beta=0.5):
    sim1=CSM_P1(doc1,doc2)
    sim2=CSM_P2(doc1,doc2)
    simValue=alpha*max(sim1,sim2)+Beta*min(sim1,sim2)
    return simValue
def BLAB_SM(doc1,doc2):
    sim1=CSMB(doc1,doc2,alpha=0.5,Beta=0.5)# CSMB10
    return sim1
def ZSM(doc1,doc2):
    Dij=Nij=Di=Ni=Nj=Dj=0
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=a.intersection(b)# indices where both docs have non zero elements
    intersection=np.array(list(intersection))
    if len(intersection)>0:
      #  prod=np.multiply(doc1[intersection],doc2[intersection])
        Dij=np.dot(doc1[intersection],doc2[intersection])#sum(prod)
    Nij=len(intersection)
    zsm=(Nij*Dij)/(np.sum(doc1)*np.sum(doc2))
    return  zsm
def CSM_P1(doc1,doc2):
    simValue=0 
    if ((not np.any(doc1)) or (not np.any( doc2))):   
        simValue=0  
    else:
        simValue=0
        N=len(doc1)#total_features;
        a=set(np.nonzero(doc1)[0])  
        b=set(np.nonzero(doc2)[0])   
        Nab=len(a.intersection(b))
        F=len(a.union(b))-Nab
        simValue=(1-F/N)
    return simValue
def CSM_P2(doc1,doc2):
    simVal=0
    if ((not np.any(doc1)) or (not np.any( doc2))):   
        simValue=0  
    else:
        simValue=0
        a=set(np.nonzero(doc1)[0])  
        b=set(np.nonzero(doc2)[0])   
        Na=len(a)
        Nb=len(b)
        Nab=len(a.intersection(b))
        simVal=(2*Nab)/(Na+Nb)
    return simVal
def DDSMa(doc1,doc2):
    sim=1-(np.sum(np.abs(doc1-doc2))/np.sum(doc1+doc2))
    return sim
def DDSMb(doc1,doc2):
    sim =1-(np.sum(np.square(doc1-doc2))/np.sum(np.square(doc1+doc2)))
    return sim
def DDSMc(doc1,doc2):
    doc3=np.square(doc1)
    doc4=np.square(doc2)
    sim= 1-(np.sum(np.abs(doc3-doc4))/np.sum(np.square(doc1+doc2)))
    return sim
def EN1_DDSMa(doc1,doc2):
    return (DDSMa(doc1,doc2)*CSM_P1(doc1,doc2))
def EN_DDSMb(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    SF=len(list(a.intersection(b)))
    N=len(doc1)
    return (DDSMb(doc1,doc2)*(SF+1)/(N+1))
def EN_DDSMc(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    SF=len(list(a.intersection(b)))
    N=len(doc1)
    return (DDSMc(doc1,doc2)*(SF+1)/(N+1))
def BASM(doc1,doc2):
    return CSM_P2(doc1,doc2) 
def ESTB_SM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    intersection=np.array(list(a.intersection(b)))
    comp1=np.array(list(a-b))
    comp2=np.array(list(b-a))
    X,Y,D1,D2,sim=0,0,0,0,0  
    sim=0
    if len(intersection)>0:
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
        if len(comp1)>0:
            D1=np.sum(doc1[comp1])
        if len(comp2)>0:
            D2=np.sum(doc2[comp2])
        if len(intersection)>0:
            sim=1/(1+((D1/X)+(D2/Y)))
    return sim 
def ESTBSM_Sim2(doc1,doc2):
    sim= (1-ESTB_SM(doc1,doc2))*CSM_P2(doc1,doc2)
    return 1-sim
def ESTB_V1(doc1,doc2):
    return (1-ESTB_SM(doc1,doc2))+CSM_P2(doc1,doc2)
def ESTB_V2(doc1,doc2):
    return (1-ESTB_SM(doc1,doc2))*CSM_P2(doc1,doc2)*(1-ANSM(doc1,doc2))
def ESTB_V3(doc1,doc2):
    estb=1-ESTB_SM(doc1,doc2)
    ansm=ANSM(doc1,doc2)
    return ((estb+ansm)/2)*CSM_P2(doc1,doc2)
def EMX1(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    intersection=a.intersection(b)
    sim=0
    if len(intersection)>0:
        meanDiff=abs(np.mean(doc1)-np.mean(doc2))
        sim=1/(1+math.exp(-1*meanDiff))
    return sim
def EMX2(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    intersection=a.intersection(b)
    sim=0
    if len(intersection)>0:
        meanDiff=abs(np.mean(doc1)-np.mean(doc2))
        stdDiff=abs(np.std(doc1)-np.std(doc2))
        sim=1/(1+math.exp(-1*meanDiff*stdDiff))
    return sim 
def EMX7(doc1,doc2):
    sim=EMX1(doc1,doc2)
    if sim>0:
        sim=sim*(1-ESTB_SM(doc1,doc2))*CSM_P2(doc1,doc2)
    return sim 
def EMX13(doc1,doc2):
    sim=EMX2(doc1,doc2)
    if sim>0:
        sim=sim*(1-ESTB_SM(doc1,doc2))*CSM_P2(doc1,doc2)
    return sim 
def PCC_Sim2(doc1,doc2):#distance
    return (1-PCC(doc1,doc2))*CSM_P2(doc1,doc2)
def ANSM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    com=np.array(list(a.intersection(b)))
    sim=0
    if len(com)>0:
        sim=np.sum(np.sum(doc1[com]+doc2[com]))/(np.sum(doc1)+np.sum(doc2))
        
    return sim   
def ANSM_V1(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    SP=len(a.intersection(b))
    N=len(doc1)
    return 1-((1-ANSM(doc1,doc2))*((SP+1)/(N+1)))
def ANSM_V2(doc1,doc2):
    return (1-ANSM(doc1,doc2))*CSM_P2(doc1,doc2)
def ANSM_V3(doc1,doc2):
    sim= ((1-ANSM(doc1,doc2))+CSM_P2(doc1,doc2))/2
    return sim
def KL(a, b):
    return entropy(a,b)

def MSTB_SM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    intersection=np.array(list(a.intersection(b)))
    sim=0
    if len(intersection)>0:
        ints=np.sum(np.intersect1d(doc1,doc2))
        D1=np.sum(np.unique(doc1))-ints
        D2=np.sum(np.unique(doc2))-ints
        X,Y=0,0  
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
        a=np.array(list(a))
        b=np.array(list(b))
        Z1,Z2=0,0
        Z1=np.sum(doc1[a])
        Z2=np.sum(doc2[b]) 
        sim=((X*Y)/(Z1*Z2))*(1-((D1*D2)/(Z1*Z2)))
    return sim 
def STB_SM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    intersection=np.array(list(a.intersection(b)))
    comp1=a-b
    comp2=b-a   
    comp1=np.array(list(comp1))
    comp2=np.array(list(comp2))
    a=np.array(list(a))
    b=np.array(list(b))
    X,Y,D1,D2,Z1,Z2,sim=0,0,0,0,0,0,0
    if len(intersection)>0:
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
    if len(comp1)>0:
        D1=np.sum(doc1[comp1])
    if len(comp2)>0:
        D2=np.sum(doc2[comp2])
    if len(a)>0:
        Z1=np.sum(doc1[a])
    if len(b)>0:
        Z2=np.sum(doc2[b]) 
    if Z1!=0 and Z2!=0:
        sim=((X*Y)/(Z1*Z2))*(1-((D1*D2)/(Z1*Z2)))
    return sim
def DSM(doc1,doc2,var):
    lemda=1
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=a.intersection(b) # indices where both docs have non zero elements
    union=a.union(b) # indices where either doc has non zero elements
    doc1=np.array(doc1)
    doc2=np.array(doc2)
    Nstar=0
    intersection=np.array(list(intersection))
    if (len(intersection)>0):
        term1=np.exp(-1*( doc1[intersection]-doc2[intersection] )/var[intersection])
        Nstar=sum(0.5* (1+term1)) 
    else:
        Nstar=lemda* -1 
    Nunion=len(union)
    dsm=0
    if Nunion!=0:
        dsm=((Nstar/Nunion)+lemda)/(1+lemda)
   
    return dsm
def TA(doc1,doc2):
    a=np.sqrt(doc1.dot(doc1))
    b=np.sqrt(doc2.dot(doc2))
    dot=np.dot(doc1,doc2)**2
    sim=0
    if a<=b:
        sim=dot/(a*(b**3))
    else:
        sim=1-(dot/(b*(a**3)))
    return sim


######################Enhnaced Measures#################################################
def EExtendedJaccard(a,b):
    vector1=set(np.where(a!=0)[0])
    vector2=set(np.where(b!=0)[0])
    dot=len(vector1.intersection(vector2))
    sum1=len(vector1)
    sum2=len(vector2)
    denom=math.sqrt(sum1)+math.sqrt(sum2)-dot
    if(denom!=0):
        return 1.0 - (float(dot)/(denom))
    else:
        return -1
    
def EPairwise(a,b):
    percentage=1 #100 percent
    K1=np.count_nonzero(a)
    K2=np.count_nonzero(b)
    k=percentage*min(K1,K2)
    setA=set(np.argpartition(a, len(a)-1 - k)[-k:])
    setB=set(np.argpartition(b, len(b) -1- k)[-k:])
    union=np.array(list(setA.union(setB)))
    elementsA=a[union]
    elementsB=b[union]
    return distance.cosine(elementsA, elementsB)

def EJS(a, b):
    p = a/ np.sum(a)
    q = b/ np.sum(b)
    m = np.add(p,q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def EDice(doc1,doc2):
    a=set(list(np.nonzero(doc1!=0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2!=0)[0]))  #indices of non zero elements in doc2
    intr=len(a.intersection(b))
    aComp=len(a-b)
    bComp=len(b-a)
    sim=0
    if (intr+aComp+bComp)!=0:
        sim=2*(intr/(2*(intr+aComp+bComp)))
    return sim
def EExtendedDice(doc1, doc2):
    a=set(list(np.nonzero(doc1!=0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2!=0)[0]))  #indices of non zero elements in doc2
    intr=len(a.intersection(b))
    sim=(2*intr)/(len(a)**2 + len(b)**2)
    return sim
def EPDSM(doc1,doc2):
    a=set(np.nonzero(doc1!=0)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2!=0)[0])  #indices of non zero elements in doc2
    intersection=np.sum(np.minimum(doc1,doc2))
    union=np.sum(np.maximum(doc1,doc2))
    PF=len(a.intersection(b))
    M=len(doc1)
    AF=M-len(a.union(b))
    psdm=(intersection/union)*((PF+1)/(M-AF+1))
    return psdm
def esmtp(doc1,doc2,var):
    lemda=0.0001
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    intersection=a.intersection(b) # indices where both docs have non zero elements
    union=a.union(b) # indices where either doc has non zero elements
    l1=len(a-intersection) # doc1 !=0 and doc2=0
    l2=len(b-intersection) # doc1 =0 and doc2!=0
    Nstar=0
    intersection=np.array(list(intersection))
    if (len(intersection)>0):
        term1=np.exp(-1*np.square(( doc1[intersection]-doc2[intersection] )/var[intersection]))
        Nstar=np.sum(0.5* (1+term1)) +lemda* -1 *(l1+l2)
    else:
        Nstar=lemda* -1 *(l1+l2)   
    Nunion=len(intersection)+l1+l2
    smtp=0
    if Nunion!=0:
        smtp=((Nstar/Nunion)+lemda)/(1+lemda)
    return smtp
def ENSMT(doc1,doc2):
    Dij=Nij=Di=Ni=Nj=Dj=0
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    intersection=a.intersection(b)# indices where both docs have non zero elements
    a=np.array(list(a-intersection)) 
    b=np.array(list(b-intersection))
    intersection=np.array(list(intersection))
    if len(intersection)>0:
      #  prod=np.multiply(doc1[intersection],doc2[intersection])
        Dij=np.dot(doc1[intersection],doc2[intersection])#sum(prod)
    if len(a)>0:
        Di=np.sum(doc1[a])
    if len(b)>0:
        Dj=np.sum(doc2[b])
    Nij=len(intersection)
    Ni=len(a)
    Nj=len(b)
    Nsmt=0
    if (Ni*Di+Dj*Nj)!=0:
        Nsmt=(Nij*Dij)/(Ni*Di+Dj*Nj)
    return Nsmt

def EBLAB_SM(doc1,doc2):
    return 0.5*(ECSM_P1(doc1,doc2)+ECSM_P2(doc1,doc2))
    '''
    if dataset=="reuters":
        sim1=ECSMB_MinMax(doc1,doc2,alpha=0.7,Beta=0.3)#CSMB12
    else:
        sim1=ECSMB(doc1,doc2,alpha=0.9,Beta=0.1)# CSMB10
    
    return sim1
    '''
def EESTB_SM(doc1,doc2):
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    intersection= np.array(list(a.intersection(b)))
    comp1=np.array(list(a-b))
    comp2=np.array(list(b-a))
    D1=D2=sim=0                   
    if len(intersection)>0:
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
        if len(comp1)>0:
            D1=np.sum(doc1[comp1])
        if len(comp2)>0:
            D2=np.sum(doc2[comp2])
        sim=1/(1+((D1/X)+(D2/Y)))
    return sim


def ECSM_P2(doc1,doc2):
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2NSMT
    sim=0
    if (len(a)+len(b))!=0:
        sim=(2*len(a.intersection(b)))/(len(a)+len(b))
    return sim



def EEMX1(doc1,doc2):
    a=set(np.nonzero(doc1!=0)[0])  
    b=set(np.nonzero(doc2!=0)[0])
    intersection=a.intersection(b)
    intersection=a.intersection(b)
    sim=0
    if len(intersection)>0:
        meanDiff=abs(np.mean(doc1)-np.mean(doc2))
        sim=1/(1+math.exp(-1*meanDiff))
    return sim


def EEMX2(doc1,doc2):
    a=set(np.nonzero(doc1!=0)[0])  
    b=set(np.nonzero(doc2!=0)[0])
    intersection=a.intersection(b)
    sim=0
    if len(intersection)>0:
        meanDiff=abs(np.mean(doc1)-np.mean(doc2))
        stdDiff=abs(np.std(doc1)-np.std(doc2))
        sim=1/(1+math.exp(-1*meanDiff*stdDiff))
    return sim 

def EEMX7(doc1,doc2):
    sim=EEMX1(doc1,doc2)
    if sim>0:
        sim=sim*(1-EESTB_SM(doc1,doc2))*ECSM_P2(doc1,doc2)
    return sim
def EEMX13(doc1,doc2):
    sim=EEMX2(doc1,doc2)
    if sim>0:
        sim=sim*(1-EESTB_SM(doc1,doc2))*ECSM_P2(doc1,doc2)
    return sim 

def EMSTB_SM(doc1,doc2):
    a=np.nonzero(doc1 != 0)[0]  #indices of non zero elements in doc1
    b=np.nonzero(doc2 != 0)[0]  #indices of non zero elements in doc2
    newDoc1=doc1[a]
    newDoc2=doc2[b]
    intersection=np.intersect1d(a,b)
    sim=0
    if len(intersection)>0:
        ints=np.sum(np.intersect1d(doc1,doc2))
        D1=np.sum(np.unique(newDoc1))-ints
        D2=np.sum(np.unique(newDoc2))-ints 
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
        Z1=np.sum(newDoc1)
        Z2=np.sum(newDoc2) 
        sim=((X*Y)/(Z1*Z2))*(1-((D1*D2)/(Z1*Z2)))
    return sim 

def ECSM_P1(doc1,doc2):
    simValue=0
    N=len(doc1)#total_features;
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2NSMT
    Nab=len(a.intersection(b))
    F=len(a.union(b))-Nab
    simValue=(1-F/N)
    return simValue
def DDSMa(doc1,doc2):
    return 1-(np.sum(np.abs(doc1-doc2))/np.sum(doc1+doc2))
def DDSMb(doc1,doc2):
    return 1-(np.sum(np.square(doc1-doc2))/np.sum(np.square(doc1+doc2)))
def DDSMc(doc1,doc2):
    doc3=np.square(doc1)
    doc4=np.square(doc2)
    return 1-(np.sum(np.abs(doc3-doc4))/np.sum(np.square(doc1+doc2)))

def EEN1_DDSMa(doc1,doc2):
    return DDSMa(doc1,doc2)*ECSM_P1(doc1,doc2)
def EEN_DDSMb(doc1,doc2):
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    SF=len(a.intersection(b))
    N=len(doc1)
    return DDSMb(doc1,doc2)*(SF+1)/(N+1)
def EEN_DDSMc(doc1,doc2):
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    SF=len(a.intersection(b))
    sim=1-(DDSMc(doc1,doc2)*(SF+1)/(len(doc1)+1))
    return sim


def EANSM(doc1,doc2):
    a=np.nonzero(doc1 != 0)[0] #indices of non zero elements in doc1
    b=np.nonzero(doc2 != 0)[0]  #indices of non zero elements in doc2
    com=np.intersect1d(a,b)
    sim=0
    if len(com)>0:
        sim=np.sum(doc1[com]+doc2[com])/np.sum(doc1+doc2)
    return sim  
def EANSM_V3(doc1,doc2):
    return (EANSM(doc1,doc2)+ECSM_P2(doc1,doc2))/2

def EZSM(doc1,doc2):
    Dij=Nij=Di=Ni=Nj=Dj=0
    a=np.nonzero(doc1 != 0)[0]  #indices of non zero elements in doc1
    b=np.nonzero(doc2 != 0)[0]  #indices of non zero elements in doc2
   # sum1=np.sum(doc1[a])
    #sum2=np.sum(doc2[b])
    a=set(list(a))
    b=set(list(b)) 
    intersection=a.intersection(b)# indices where both docs have non zero elements
    intersection=np.array(list(intersection))
    Nij=len(intersection)
    if Nij>0:
        Dij=np.dot(doc1[intersection],doc2[intersection])#sum(prod)
    zsm=(Nij*Dij)/(np.sum(doc1)*np.sum(doc2))
    return zsm
def EEnhancedJaccard(a, b):#similarity
    a=set(np.nonzero(a!=0)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(b!=0)[0])  #indices of non zero elements in doc2
    union=a.union(b)
    intersection=a.intersection(b)
    sim=0
    if len(union)!=0:
        sim=len(intersection)/len(union)#similarity
    return sim
def esimIT(a,b):
    p1= np.divide(a,termOcc) 
    p2= np.divide(b,termOcc)
    minVal=np.minimum(p1,p2)
    pi=np.log((np.array(docOcc)/totalDocs))
    sIT=(2*np.sum(np.multiply(minVal,pi)))/((np.sum(np.multiply(p1,pi)))+(np.sum(np.multiply(p2,pi))))
    return sIT
def STB_SM_new(doc1,doc2):
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    intersection=np.array(list(a.intersection(b)))
    comp1=a-b
    comp2=b-a   
    comp1=np.array(list(comp1))
    comp2=np.array(list(comp2))
    X,Y,D1,D2,Z1,Z2,sim=0,0,0,0,0,0,0
    if len(intersection)>0:
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
    if len(comp1)>0:
        D1=np.sum(doc1[comp1])
    if len(comp2)>0:
        D2=np.sum(doc2[comp2])
    Z1=np.sum(doc1)
    Z2=np.sum(doc2) 
    if Z1!=0 and Z2!=0:
        sim=((X*Y)/(Z1*Z2))*(1-((D1*D2)/(Z1*Z2)))
    return sim
def EBASM(doc1,doc2):
    return ECSM_P2(doc1,doc2) 
def EISC(doc1,doc2): 
    dot=np.sum(np.sqrt(np.multiply(doc1,doc2)))
    isc=dot /( math.sqrt(np.linalg.norm(doc1, ord=1))*math.sqrt(np.linalg.norm(doc2, ord=1)))
    return isc

#SP equations
def ESP(doc1,doc2):
    a=set(list(np.nonzero(doc1!=0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2!=0)[0]))  #indices of non zero elements in doc2
    nonzeroTerm=list(a.intersection(b))  #indices where both docs have non zero elemments
    d1=doc1[nonzeroTerm] #doc1 values of doc1 intersection  doc2
    d2=doc2[nonzeroTerm] #doc2 values of doc1 intersection  doc2
    minArr=np.minimum(d1,d2) #minimum values  of d1 and d2
    maxArr=np.maximum(d1,d2) #maximum values  of d1 and d2
    ln=len(nonzeroTerm)
    docCount=np.zeros(ln)
    aData=allData[:,nonzeroTerm] #extracting data with only feature indices of doc1 intersection  doc2
    for ti in range(0,ln):
        docCount[ti]=count_values_in_range(aData[:,ti], minArr[ti], maxArr[ti]) #count values which lies between min and max
    #norm factor which is ist part of equation 6 in paper
    normFactor= len(a.union(b))#norm factor equal to length of doc1 union doc2
    dCount=np.array(docCount[np.nonzero(docCount!=0)[0]])# get all non zero values between min and max
    SP_val=0
    if(normFactor!=0):
        SP_val=np.sum(np.log(totalDocs/dCount)) #2nd part in equation 6 in paper
        SP_val = (SP_val/normFactor) #equation 6 in paper
    return SP_val
def ENSMT_BASM(doc1,doc2):
    return ENSMT(doc1,doc2)*BASM(doc1,doc2)
    
def PCC_BASM(doc1,doc2):
    return PCC(doc1,doc2)*BASM(doc1,doc2)


class SimMeaure:

    #constructor

    def __init__(self,k = 1,metric="euclidean",termOccurance=None, docOccurance=None):

        
        self.k=k
        #print ("KNN",self.k)
        self.metric=metric
        self.trainingData=[]
        self.trainLabels=[]
        self.smtp=None
        self.termOccurance=termOccurance
        self.docOccurance=docOccurance
         
    
         

    def fit(self, training_data, trainLabels ):
        self.trainingData=training_data
        self.trainLabels=trainLabels
        
            
       
    def predict(self, testData):
        train=self.trainingData
        var=np.zeros(train.shape[1])
        if(self.metric=="smtp" or  self.metric=="esmtp" or  self.metric=="DSM"):
            var=np.var(train,axis=0)
            print("Vrainace is",var)
        distList=["KL","extendedJaccard","bhatta","Euclidean","Cosine","Jaccard","JS","EJS","Dice",
                  "Pairwise","EPairwise","Manhattan","EnhancedJaccard","EEnhancedJaccard","eextendedJaccard","EJS","Minkowski"]#distance
        func=globals()[self.metric]
        predLabel=[]
        kList=[1,3,5,9,15,30,45,70,90,120]
        for kk in range(0,len(kList)):
            temp=[]
            predLabel.append(temp)
            ''''''
        distMatrix=[]         
        for i in(range(0,len(testData))):
            #print(i)
            dist=[]
            for j in(range(0,len(train))):
                if(self.metric=="smtp" or self.metric=="smtp_improved" or self.metric=="esmtp" or self.metric=="DSM"):
                    dist.append(func(testData[i],train[j],var)) 
                elif(self.metric=="simIT" or self.metric=="esimIT"  ):
                    dist.append(simIT(testData[i], train[j],self.termOccurance,self.docOccurance))
                elif(self.metric=="SP"):
                    doc1=testData[i]
                    doc2=train[j]
                    nonzeroDoc1=set(np.nonzero(doc1)[0])
                    nonzeroDoc2=set(np.nonzero(doc2)[0])
                    nonzeroTerm=list(nonzeroDoc1.intersection(nonzeroDoc2))
                    minArr=np.minimum(doc1[nonzeroTerm],doc2[nonzeroTerm])
                    maxArr=np.maximum(doc1[nonzeroTerm],doc2[nonzeroTerm])
                    ln=len(nonzeroTerm)
                    aData=allData[:,nonzeroTerm]
                    docCount=np.zeros(ln)
                    for ti in range(0,ln):
                        tData=pd.Series(aData[:,ti])
                        docCount[ti]=count_values_in_range(tData, minArr[ti], maxArr[ti])
                    dist.append(SP(testData[i],train[j],docCount,totalDocs))
                else:
                    dist.append(func(testData[i], train[j]))
            flag=-1
            dist=list(dist)
            if(self.metric in distList):
                
                dist=[(1/(x+0.0001)) for x in dist]
                flag=1

            distMatrix.append(dist)
        return distMatrix


def groupData(labels,categories):

    print (categories)
    groups=[]
    for i in range(0,len(categories)):
        groups.append([0,0,0])
    totalDocs=len(labels)
    print (totalDocs)
    for i in range(0,totalDocs):
        tmp=(categories.index(labels[i]))
        groups[tmp].append(i)
    for i in range(0,len(categories)):
        del (groups[i])[0:3]
    return groups

class ANN() : # to take preprocessed data , pass them into model and print results
    def __init__ (self ,  x_train  , x_test , y_train , y_test ):
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
    def fit(self,x_train  , y_train  , x_test, y_test , outputdim) :         
      #Fitting ANN Model
        config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 6} ) 
        sess = tf.compat.v1.Session(config=config) 
        K.set_session(sess)
        model = Sequential()
        input_dim = x_train.shape[1]  # Number of features
        #micro_f1 = metrics.F1Score(num_classes=input_dim , average='micro')
        y_train = np.asarray(train_labels)
        y_test = np.asarray(test_labels)
        model.add(Dense(500, activation='tanh',kernel_regularizer='l2',input_shape = (input_dim,)))
        model.add(Dense(100, activation='tanh',kernel_regularizer='l2'))
        #model.add(Dense(20, activation='relu',kernel_regularizer='l2'))
        model.add(Dense(outputdim, activation='softmax'))
        model.summary()
        adm=RMSprop(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adm ,metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                    epochs=92,
                    validation_data=(x_test, y_test),
                     verbose=1)
         #Finally testing the model & getting accuracy metrics
        model.summary()
        loss,acc = model.evaluate(x_test, y_test, verbose=1)
        y_pred1 = model.predict(x_test)
        predicts = np.max(y_pred1,axis=1) 
        sess.close()
        gc.collect()
 
        return np.argmax(y_pred1,axis=1) ,y_pred1
def ECSM_P1(doc1,doc2):
    simValue=0
    N=len(doc1)#total_features;
    a=set((np.nonzero(doc1 != 0)[0]).tolist()) #indices of non zero elements in doc1
    b=set((np.nonzero(doc2 != 0)[0]).tolist()) #indices of non zero elements in doc2NSMT
    Nab=len(a.intersection(b))
    F=len(a.union(b))-Nab
    simValue=(1-F/N)
    return simValue
def ECSM_P2(doc1,doc2):
    a=set((np.nonzero(doc1 != 0)[0]).tolist()) #indices of non zero elements in doc1
    b=set((np.nonzero(doc2 != 0)[0]).tolist()) #indices of non zero elements in doc2NSMT
    sim=0
    if (len(a)+len(b))!=0:
        sim=(2*len(a.intersection(b)))/(len(a)+len(b))
    return sim

def EBLAB_SM(doc1,doc2):
    return 0.5*(ECSM_P1(doc1,doc2)+ECSM_P2(doc1,doc2))

def ESTB_SM(doc1,doc2):
    a=set(np.nonzero(doc1)[0])  
    b=set(np.nonzero(doc2)[0])
    intersection=np.array(list(a.intersection(b)))
    comp1=np.array(list(a-b))
    comp2=np.array(list(b-a))
    X,Y,D1,D2,sim=0,0,0,0,0                   
    if len(intersection)>0:
        X=np.sum(doc1[intersection])
        Y=np.sum(doc2[intersection])
        if len(comp1)>0:
            D1=np.sum(doc1[comp1])
        if len(comp2)>0:
            D2=np.sum(doc2[comp2])
        if len(intersection)>0:
            sim=1/(1+((D1/X)+(D2/Y)))
    return sim 
def ESP(doc1,doc2):
    a=set(list(np.nonzero(doc1!=0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2!=0)[0]))  #indices of non zero elements in doc2
    nonzeroTerm=list(a.intersection(b))  #indices where both docs have non zero elemments
    d1=doc1[nonzeroTerm] #doc1 values of doc1 intersection  doc2
    d2=doc2[nonzeroTerm] #doc2 values of doc1 intersection  doc2
    minArr=np.minimum(d1,d2) #minimum values  of d1 and d2
    maxArr=np.maximum(d1,d2) #maximum values  of d1 and d2
    ln=len(nonzeroTerm)
    docCount=np.zeros(ln)
    aData=allData[:,nonzeroTerm] #extracting data with only feature indices of doc1 intersection  doc2
    for ti in range(0,ln):
        docCount[ti]=count_values_in_range(aData[:,ti], minArr[ti], maxArr[ti]) #count values which lies between min and max
    #norm factor which is ist part of equation 6 in paper
    normFactor= len(a.union(b))#norm factor equal to length of doc1 union doc2
    dCount=np.array(docCount[np.nonzero(docCount!=0)[0]])# get all non zero values between min and max
    SP_val=0
    if(normFactor!=0):
        SP_val=np.sum(np.log(totalDocs/dCount)) #2nd part in equation 6 in paper
        SP_val = (SP_val/normFactor) #equation 6 in paper
    return SP_val

def esmtp(doc1,doc2,var):
    lemda=1
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2
    intersection=a.intersection(b) # indices where both docs have non zero elements
    union=a.union(b) # indices where either doc has non zero elements
    l1=len(a-intersection) # doc1 !=0 and doc2=0
    l2=len(b-intersection) # doc1 =0 and doc2!=0
    Nstar=0
    intersection=np.array(list(intersection))
    if (len(intersection)>0):
        term1=np.exp(-1*np.square(( doc1[intersection]-doc2[intersection] )/var[intersection]))
        Nstar=np.sum(0.5* (1+term1)) +lemda* -1 *(l1+l2)
    else:
        Nstar=lemda* -1 *(l1+l2)   
    Nunion=len(intersection)+l1+l2
    smtp=((Nstar/Nunion)+lemda)/(1+lemda)
    return smtp
def classify(metric='Cosine',termOccurance=None, docOccurance=None):
    time1= time.time()
    accuracy=[]
    macroFMeasure=[]
    microFMeasure=[]
    
    y_pred =NBCombined(train_data, train_labels,test_data,test_labels,metric)
    time2= time.time()
    categories=list(set(train_labels))
    
    for i in range(0,1):
        #Accuracy
    
        accuracy=accuracy_score(test_labels, y_pred)
        
        #Macro Precision
        
        macroP=precision_score(test_labels, y_pred,average='macro')
        
        #Micro Precision
        
        microP=precision_score(test_labels, y_pred,average='micro')
        
        #Macro Recall
        
        macroR=recall_score(test_labels, y_pred,average='macro')
        
        #Micro Recall
        
        microR=recall_score(test_labels, y_pred,average='micro')
        
        
        fMeasure=f1_score(test_labels, y_pred,average='macro')
       # PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="F Measure",Arr=fMeasure)
        
        m_fMeasure=f1_score(test_labels, y_pred,average='micro')
       # PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="F Measure",Arr=fMeasure)
        
        #PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="g Measure",Arr=gMeasure)
      
        test = label_binarize(test_labels, classes=categories)
        pred = label_binarize( y_pred, classes=categories)
        
        #roc
        roc=roc_auc_score(test, pred)
    #Accuracy
    accuracy_avg.append(accuracy)
    #Precision
    macroP_avg.append(macroP)
    microP_avg.append(microP)
    #recall
    macroR_avg.append(macroR)
    microR_avg.append(microR)
    #F measure
    macroFMeasure_avg.append(fMeasure)
    microFMeasure_avg.append( m_fMeasure)
    #Roc
    roc_avg.append(roc)

def PrintDetails(metric="smtp",time=None,measure= "Accuracy",Arr=None):
    x = PrettyTable()
    kList=[1,3,5,9,15,30,45]
    x.field_names = ["Experiment No.","DataSet1"]
    print("metric",metric)
    print("time",time)
    print("measure",measure)
    tables.write("metric\t"+str(metric)+"\n")
    tables.write("time\t"+str(time)+"\n")
    tables.write("measure\t"+str(measure)+"\n")
    average=0
    for i in range(0,len(Arr)):
        x.add_row([i+1,Arr[i]]) 
        average+=Arr[i]
    x.add_row(["Average",average/len(Arr)])   
    tables.write(str(x))
    print (x)
def NBCombined(train_data, train_labels,test_data,test_labels,metric):
    #prediction of NB on test data
    classes=list(set(train_labels))
    global prob
    if len(prob)==0:
        cnn = ANN(train_data, train_labels,test_data,test_labels)
        y_pred,prob = cnn.fit(train_data, train_labels,test_data,test_labels,len(classes))
        #y_pred = np.max(y_pred1,axis=1)
    if metric=="Base":
        return y_pred

    #y_pred = classifier.predict(test_data)
    classifier =SimMeaure(k=1,metric=metric)#,termOccurance=termOccurance,docOccurance=docOccurance)  
    classifier.fit(train_data, train_labels)
    distMatrix = classifier.predict(test_data) 
    print("Dist Mat shape", len(distMatrix))
    print("Element shape", len(distMatrix[0]))
    distList=[]
    predLabels=[]
    nn=10
    prob=np.array(prob)
    print(prob.shape)
    for tst in range(0,test_data.shape[0]):
        probNb=np.array(prob[tst,:])
        simAvg=[]
        simValue=np.array(distMatrix[tst])
        neigh= np.argpartition(np.array(simValue), len(simValue) - nn)[-nn:]
        neigh_labels=np.array(train_labels)[neigh]
        for cl in range(0,len(classes)):
            avg=0
            indices=[l for l in range(0,len(neigh_labels)) if neigh_labels[l]==classes[cl]]
            if len(indices)>0:
                avg=np.nanmean(simValue[indices])
            #print(avg)
            simAvg.append(avg)
        #simAvg=(np.array(simAvg))/nn
        if (np.nanmax(simAvg)!=0):
            simAvg=(simAvg/np.nanmax(simAvg))
        #print(simAvg)
        #print(probNb)
        totalValue=np.add(probNb, simAvg)
       # totalValue=simAvg
        #print("combined value is",totalValue)
        lblInd=np.where(totalValue == np.max(totalValue))
        ind1=0
        if len(lblInd[0])>0:
            ind1=(lblInd[0])[0]
       # print(ind1)
       # print(lbl)
        predLabels.append(ind1)      

    return predLabels
dataset='20ng'
feat_dict, label_dict,precompute_time=loadData(dataset)
print(type(feat_dict['train']))
#arr=features
train_data=(feat_dict['train']).cpu().detach().numpy()
test_data=(feat_dict['test']).cpu().detach().numpy()
train_labels=(label_dict['train']).cpu().detach().numpy()
test_labels=(label_dict['test']).cpu().detach().numpy()

#allData=np.array(arr)
'''
,"Euclidean","Manhattan","Cosine","Jaccard","bhatta","EPairwise",          
          "EPDSM","PCC","STB_SM_new","esmtp","EISC","EDice","ESP","EBLAB_SM"

'''
measures=["Base"]
prob=[]
termOcc,docOcc=[],[]
for met in measures:
    time1=time.time()
    accuracy_avg=[]
    macroP_avg=[]
    microP_avg=[]
    macroR_avg=[]
    microR_avg=[]
    macroFMeasure_avg=[]
    microFMeasure_avg=[]
    roc_avg=[]
    if  met=="esimIT":
        termOcc=np.sum(allData,axis=0)
        docOcc=np.count_nonzero(allData,axis=0)
    print("###############")
    fname='table_'+dataset+'_ANN_'+met+'_sgc_5RE.txt'
    tables = open(fname, 'w')
    #tables.write("No. of features\t"+str(n)+"\n")
    print("###############")
    for i in range(0,5):
        prob=[]
        classify(metric=met)
    #print ("nTerms",n)
    time2=time.time()
    #Accuracy
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="Accuracy",Arr=accuracy_avg)
    #Precision
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="Macro Precision",Arr=macroP_avg)
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="Micro Precision",Arr=microP_avg)
    #Recall
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="Macro Recall",Arr=macroR_avg)
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="Micro Recall",Arr=microR_avg)
    #f measure
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="F Measure",Arr=macroFMeasure_avg)

    #f measure
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="Micro F Measure",Arr=microFMeasure_avg)

    #roc
    PrintDetails(metric=met,time=str(timedelta(seconds=(time2-time1))),measure="ROC",Arr=roc_avg)
    tables.close()
    print("###############")







# In[ ]:





# In[2]:


totalValue=[0, np.nan, np.nan, 0]
np.where(totalValue == np.nanmax(totalValue))


# In[ ]:




