#!/usr/bin/env python
# coding: utf-8

# **Libraries**

# In[1]:


import pandas as pd
import numpy as np

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import joypy #el de las densidades guays

from sklearn import manifold  
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import os 
cwd = os.getcwd() + "/"
cwd ="/home/angela/Notebook/machine_learning/normalizados/btag_sis_error/"
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from IPython.display import display

from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore') #ATENCION QUE ESTO CREO QUE FUNCIONA PARA TODO EL NOTEBOOK

import random
random.seed(6)
np.random.seed(6)
np.random.RandomState(6)

from sklearn.preprocessing import MinMaxScaler
import pickle


# Si suponemos que el 65 % es lo que conseguimos sin forzar mucho el método ( me refiero en el análisis de física en ATLAS) , podemos fijarlo en 65%
# ese 35% puedes ponderarlo con las proporciones que has obtenido: así tendremos 13*35/13.6 para 1 b tag y 0.6*35/13.6 para 0 tag , ¿lo ves?
# Porque en datos toy simulados reconstruimos sin ambigüedades, tenemos los 4 jets más energéticos (es un caso casi ideal) , datos simulado detallados y en dato reales hay más mala concordancia entre lo que se reconstruye e identifica y lo que es 'truth'
# En el proceso de reconsturcción de jets y de elegir jets que sean los del esquema de decay de los tops nos estamos equivocando ( mismatch).
# ampliando lo fake hasta el 35% estás siendo más realista

# **LOAD**

# In[2]:


#ALL THE DATASET

df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainpickle")
df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testpickle")


# In[3]:


#SIMPLe DATASET
#df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainsimplepickle").sample(n=1000, random_state=1)
#df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testsimplepickle").sample(n=300, random_state=1)


# In[4]:


df.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1).value_counts()/df.shape[0]*100


# In[5]:


cwd2=cwd+"train2b"

df_originaltest=df_originaltest[df_originaltest.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2]

print("test bien y variando el train")

print("B_tag_train_prop en test", "\n" ,df_originaltest.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1).value_counts()/df_originaltest.shape[0]*100)

print("test", pd.crosstab(df_originaltest['label'],df_originaltest['mass']))


# # Machine learning

# **We will follow the scikit-learn schema to train and validate the model**
# 
# 
# https://scikit-learn.org/stable/modules/cross_validation.html
# 

# ## RF 

# In[6]:


#Scores


def Scores(y_true,y_pred):
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    prec=tp / (tp + fp)
    recall= tp / (tp + fn)
    F1_score= 2 * (prec * recall) / (prec + recall)
    acc=metrics.accuracy_score(y_true,y_pred)
    kappa_cohen=metrics.cohen_kappa_score(y_true,y_pred)
        
    return(tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen)


def model(X_train,y_train,X_test,rs=0):
    
    scaler = MinMaxScaler()

    
    
    model = RandomForestClassifier()
    #parámetros fijos: 
    params = {'n_estimators':[500],'min_samples_leaf':[1],'max_features':['sqrt'],'criterion':['entropy'], 'random_state':[rs]}

    

    grid = GridSearchCV(estimator=model, param_grid=params,cv=5,verbose=1, n_jobs=-1)

    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_
    
    best_model.fit(X_train,y_train)
    # Predict test set labels
    y_pred = best_model.predict(X_test)    
    y_pred_proba = best_model.predict_proba(X_test)[::,1] #Neccesary to make the ROC curve 

    return(grid,best_model,y_pred,y_pred_proba)


# ## Carga

# In[7]:



cwd2=cwd+"train2b"

#cargo otra vez para resetear el df
df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainpickle")
df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testpickle")
df_originaltest=df_originaltest[df_originaltest.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2]

print("test bien y variando el train")

print("B_tag_train_prop", "\n" ,df_originaltest.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1).value_counts()/df_originaltest.shape[0]*100)

print("test", pd.crosstab(df_originaltest['label'],df_originaltest['mass']))


# ## Test constante, train constante, semilla cte
# 

# In[8]:


all0=[]
all1=[]
all2=[]
orig=[]
real=[]
 
#fijo la semilla:: 
rs=0
    
for btag in ["all0","all1","all2","orig","real",]:

    df_metrics = pd.DataFrame(index=[500,750,1000,1250,1500], columns=["tn", "fp", "fn", "tp", "acc", "prec","recall",
                                                                   "F1_score","kappa_cohen","auc"])
    print("BTAG =", btag)

    df_btag=df.copy()

    if btag=="all0":
        df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==0]
    elif btag=="all1":
        df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==1]
        df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)
    elif btag=="all2":                          
        df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2]
        df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)

    elif btag=="orig":                          
        df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)

    elif btag=="real": #65,33,2
        df0=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==0]
        df1=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==1].sample(n=int(df0.shape[0]*(33/2)), random_state=6, replace=False)
        df2=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2].sample(n=int(df0.shape[0]*(65/2)), random_state=6, replace=False)
        frames = [df1, df2, df0]
        df_btag = pd.concat(frames)
        df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)


    for i in (500,750,1000,1250,1500):

        print('mass=', i )

        dfmass1000=df_btag.loc[df_btag['mass'] == i]
        dfmass0=df_btag.loc[df_btag['mass'] == 0].sample(n=dfmass1000.shape[0], random_state=rs, replace=False)        #0 same size of dfmass1000
        dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(frac=1).reset_index(drop=True)    #concatenating and shuffling
        dfmass1000=dfmass1000.drop('mass', axis=1) 

        #test


        #test
        dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
        dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(random_state=rs,n=dfmass1000test.shape[0])
        np.random.seed(rs)
        dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
        dfmass1000test=dfmass1000test.drop('mass', axis=1)



        dfmass1000=dfmass1000.sample(n=9300, random_state=rs)
        dfmass1000test=dfmass1000test.sample(n=4650, random_state=rs)

        print("B_tag_train_prop", "\n" ,df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1).value_counts()/df_btag.shape[0]*100)


        drop=['lep_eta', 'lep_phi',  'met_phi', 'jets_no', 'jet1_eta', 'jet1_phi', 'jet1_btag', 
           'jet2_eta', 'jet2_phi', 'jet2_btag',  'jet3_eta', 'jet3_phi',
           'jet3_btag', 'jet4_eta', 'jet4_phi', 'jet4_btag']

        X_train=dfmass1000.drop(['label'], axis=1)
        X_train=X_train.drop(drop, axis=1)
        y_train=dfmass1000.label

        X_test=dfmass1000test.drop(['label'], axis=1)
        X_test=X_test.drop(drop, axis=1)
        y_test=dfmass1000test.label

        print("X_train_shape",X_train.shape)
        print("X_test_shape",X_test.shape)




        grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test)
        print(best_model)


    #Roc curve construction

        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        #plt.savefig(cwd2+'MASS=' + str(i) + 'ROC.png')
        plt.show()
        plt.clf() 

    #METRICS 

        tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen=Scores(y_test,y_pred)

        df_metrics.tn[i]=tn
        df_metrics.fp[i]=fp
        df_metrics.fn[i]=fn
        df_metrics.tp[i]=tp
        df_metrics.acc[i]=acc
        df_metrics.prec[i]=prec
        df_metrics.recall[i]=recall
        df_metrics.F1_score[i]=F1_score
        df_metrics.kappa_cohen[i]=kappa_cohen
        df_metrics.auc[i]=auc


    display(df_metrics)
    eval(btag).append(df_metrics)


# In[9]:


#esto es para que pueda hacer medias, std y no se ralle , no sirve para nada más 
all0.append(all0[0]) 
all1.append(all1[0])
all2.append(all2[0])
orig.append(orig[0])
real.append(real[0])


# In[10]:


l=[all0,all1,all2,orig,real]

with open("rf_traincte_semcte.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)
    
with open("rf_traincte_semcte.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


# In[11]:


with open("rf_traincte_semcte.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


# In[12]:


dfall0=pd.DataFrame()
dfall1=pd.DataFrame()
dfall2=pd.DataFrame()
orig=pd.DataFrame()
real=pd.DataFrame()

ldf=[dfall0,dfall1,dfall2,orig,real]

ldfmean=ldf.copy()
ldfstd=ldf.copy()



for j in range (5):
    ldf[j]=b[j][0]
    for i in range(len(b[j])):
        if i!=0:
            ldf[j] = pd.concat([ldf[j], b[j][i]]).apply(pd.to_numeric)
            
    ldfmean[j]=ldf[j].groupby(level=0).mean() 
    ldfstd[j]=ldf[j].groupby(level=0).std()     
    
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(["500","750","1000","1250","1500"], ldfmean[0].F1_score, label="0 jets b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[1].F1_score, label="1 jet b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[2].F1_score, label="2 jets b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[3].F1_score, label="Proporción jets b: 86%-13%-1%")
ax.plot(["500","750","1000","1250","1500"], ldfmean[4].F1_score, label="Proporción jets b: 65%-33%-2%")

ax.fill_between(["500","750","1000","1250","1500"], ldfmean[0].F1_score - ldfstd[0].F1_score, ldfmean[0].F1_score + ldfstd[0].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[1].F1_score - ldfstd[1].F1_score, ldfmean[1].F1_score + ldfstd[1].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[2].F1_score - ldfstd[2].F1_score, ldfmean[2].F1_score + ldfstd[2].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[3].F1_score - ldfstd[3].F1_score, ldfmean[3].F1_score + ldfstd[3].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[4].F1_score - ldfstd[4].F1_score, ldfmean[4].F1_score + ldfstd[4].F1_score, alpha=0.3)



ax.set_xlabel('Masa de test', fontsize=11)
ax.set_ylabel('F1-score', fontsize=11)
ax.legend(fontsize='large',loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle('Comparación de F1-score para las diferentes configuraciones de b-tag' , fontsize=14)
plt.show()


#  ## Test constante, train cte, semilla variable

# In[13]:


all0=[]
all1=[]
all2=[]
orig=[]
real=[]

rs=0

    
for rssem in range(10):
    
    for btag in ["all0","all1","all2","orig","real",]:

        df_metrics = pd.DataFrame(index=[500,750,1000,1250,1500], columns=["tn", "fp", "fn", "tp", "acc", "prec","recall",
                                                                       "F1_score","kappa_cohen","auc"])
        print("BTAG =", btag)

        df_btag=df.copy()

        if btag=="all0":
            df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==0]
        elif btag=="all1":
            df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==1]
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)
        elif btag=="all2":                          
            df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2]
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)

        elif btag=="orig":                          
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)

        elif btag=="real": #65,33,2
            df0=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==0]
            df1=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==1].sample(n=int(df0.shape[0]*(33/2)), random_state=6, replace=False)
            df2=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2].sample(n=int(df0.shape[0]*(65/2)), random_state=6, replace=False)
            frames = [df1, df2, df0]
            df_btag = pd.concat(frames)
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)


        for i in (500,750,1000,1250,1500):

            print('mass=', i )

            dfmass1000=df_btag.loc[df_btag['mass'] == i]
            dfmass0=df_btag.loc[df_btag['mass'] == 0].sample(n=dfmass1000.shape[0], random_state=rs, replace=False)        #0 same size of dfmass1000
            dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(frac=1).reset_index(drop=True)    #concatenating and shuffling
            dfmass1000=dfmass1000.drop('mass', axis=1) 

            #test
            #Siempre el mismo test: !!!
            dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
            dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(random_state=0,n=dfmass1000test.shape[0])
            np.random.seed(0)
            dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
            dfmass1000test=dfmass1000test.drop('mass', axis=1)
            np.random.seed(rs)
            dfmass1000=dfmass1000.sample(n=9300, random_state=rs)
            dfmass1000test=dfmass1000test.sample(n=4650, random_state=0)



            print("B_tag_train_prop", "\n" ,df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1).value_counts()/df_btag.shape[0]*100)


            drop=['lep_eta', 'lep_phi',  'met_phi', 'jets_no', 'jet1_eta', 'jet1_phi', 'jet1_btag', 
               'jet2_eta', 'jet2_phi', 'jet2_btag',  'jet3_eta', 'jet3_phi',
               'jet3_btag', 'jet4_eta', 'jet4_phi', 'jet4_btag']

            X_train=dfmass1000.drop(['label'], axis=1)
            X_train=X_train.drop(drop, axis=1)
            y_train=dfmass1000.label

            X_test=dfmass1000test.drop(['label'], axis=1)
            X_test=X_test.drop(drop, axis=1)
            y_test=dfmass1000test.label

            print("X_train_shape",X_train.shape)
            print("X_test_shape",X_test.shape)




            grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test,rssem)
            print(best_model)


        #Roc curve construction

            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
            plt.legend(loc=4)
            #plt.savefig(cwd2+'MASS=' + str(i) + 'ROC.png')
            plt.show()
            plt.clf() 

        #METRICS 

            tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen=Scores(y_test,y_pred)

            df_metrics.tn[i]=tn
            df_metrics.fp[i]=fp
            df_metrics.fn[i]=fn
            df_metrics.tp[i]=tp
            df_metrics.acc[i]=acc
            df_metrics.prec[i]=prec
            df_metrics.recall[i]=recall
            df_metrics.F1_score[i]=F1_score
            df_metrics.kappa_cohen[i]=kappa_cohen
            df_metrics.auc[i]=auc


        display(df_metrics)
        eval(btag).append(df_metrics)


# In[14]:


l=[all0,all1,all2,orig,real]

with open("rf_traincte_semvar.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)
    
with open("rf_traincte_semvar.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


# In[15]:


with open("rf_traincte_semvar.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


# In[16]:


dfall0=pd.DataFrame()
dfall1=pd.DataFrame()
dfall2=pd.DataFrame()
orig=pd.DataFrame()
real=pd.DataFrame()

ldf=[dfall0,dfall1,dfall2,orig,real]

ldfmean=ldf.copy()
ldfstd=ldf.copy()



for j in range (5):
    ldf[j]=b[j][0]
    for i in range(len(b[j])):
        if i!=0:
            ldf[j] = pd.concat([ldf[j], b[j][i]]).apply(pd.to_numeric)
            
    ldfmean[j]=ldf[j].groupby(level=0).mean() 
    ldfstd[j]=ldf[j].groupby(level=0).std()        


# In[17]:


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(["500","750","1000","1250","1500"], ldfmean[0].F1_score, label="0 jets b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[1].F1_score, label="1 jet b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[2].F1_score, label="2 jets b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[3].F1_score, label="Proporción jets b: 86%-13%-1%")
ax.plot(["500","750","1000","1250","1500"], ldfmean[4].F1_score, label="Proporción jets b: 65%-33%-2%")

ax.fill_between(["500","750","1000","1250","1500"], ldfmean[0].F1_score - ldfstd[0].F1_score, ldfmean[0].F1_score + ldfstd[0].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[1].F1_score - ldfstd[1].F1_score, ldfmean[1].F1_score + ldfstd[1].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[2].F1_score - ldfstd[2].F1_score, ldfmean[2].F1_score + ldfstd[2].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[3].F1_score - ldfstd[3].F1_score, ldfmean[3].F1_score + ldfstd[3].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[4].F1_score - ldfstd[4].F1_score, ldfmean[4].F1_score + ldfstd[4].F1_score, alpha=0.3)



ax.set_xlabel('Masa de test', fontsize=11)
ax.set_ylabel('F1-score', fontsize=11)
ax.legend(fontsize='large',loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle('Comparación de F1-score para las diferentes configuraciones de b-tag' , fontsize=14)
plt.show()


#  ## Test constante, train variable, semilla variable
# 
# 

# In[18]:


all0=[]
all1=[]
all2=[]
orig=[]
real=[]
    
for rs in range(10):
    
    for btag in ["all0","all1","all2","orig","real",]:

        df_metrics = pd.DataFrame(index=[500,750,1000,1250,1500], columns=["tn", "fp", "fn", "tp", "acc", "prec","recall",
                                                                       "F1_score","kappa_cohen","auc"])
        print("BTAG =", btag)

        df_btag=df.copy()

        if btag=="all0":
            df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==0]
        elif btag=="all1":
            df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==1]
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)
        elif btag=="all2":                          
            df_btag=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2]
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)

        elif btag=="orig":                          
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)

        elif btag=="real": #65,33,2
            df0=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==0]
            df1=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==1].sample(n=int(df0.shape[0]*(33/2)), random_state=6, replace=False)
            df2=df_btag[df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1)==2].sample(n=int(df0.shape[0]*(65/2)), random_state=6, replace=False)
            frames = [df1, df2, df0]
            df_btag = pd.concat(frames)
            df_btag=df_btag.sample(df_btag.shape[0], random_state=rs).reset_index(drop=True)


        for i in (500,750,1000,1250,1500):

            print('mass=', i )

            dfmass1000=df_btag.loc[df_btag['mass'] == i]
            dfmass0=df_btag.loc[df_btag['mass'] == 0].sample(n=dfmass1000.shape[0], random_state=rs, replace=False)        #0 same size of dfmass1000
            dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(frac=1).reset_index(drop=True)    #concatenating and shuffling
            dfmass1000=dfmass1000.drop('mass', axis=1) 

            #test
            #Siempre el mismo test: !!!
            dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
            dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(random_state=0,n=dfmass1000test.shape[0])
            np.random.seed(0)
            dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
            dfmass1000test=dfmass1000test.drop('mass', axis=1)
            np.random.seed(rs)
            dfmass1000=dfmass1000.sample(n=9300, random_state=rs)
            dfmass1000test=dfmass1000test.sample(n=4650, random_state=0)




            print("B_tag_train_prop", "\n" ,df_btag.loc[:,['jet1_btag','jet2_btag',"jet3_btag","jet4_btag"]].sum(axis=1).value_counts()/df_btag.shape[0]*100)


            drop=['lep_eta', 'lep_phi',  'met_phi', 'jets_no', 'jet1_eta', 'jet1_phi', 'jet1_btag', 
               'jet2_eta', 'jet2_phi', 'jet2_btag',  'jet3_eta', 'jet3_phi',
               'jet3_btag', 'jet4_eta', 'jet4_phi', 'jet4_btag']

            X_train=dfmass1000.drop(['label'], axis=1)
            X_train=X_train.drop(drop, axis=1)
            y_train=dfmass1000.label

            X_test=dfmass1000test.drop(['label'], axis=1)
            X_test=X_test.drop(drop, axis=1)
            y_test=dfmass1000test.label

            print("X_train_shape",X_train.shape)
            print("X_test_shape",X_test.shape)




            grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test, rs)
            print(best_model)


        #Roc curve construction

            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
            plt.legend(loc=4)
            #plt.savefig(cwd2+'MASS=' + str(i) + 'ROC.png')
            plt.show()
            plt.clf() 

        #METRICS 

            tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen=Scores(y_test,y_pred)

            df_metrics.tn[i]=tn
            df_metrics.fp[i]=fp
            df_metrics.fn[i]=fn
            df_metrics.tp[i]=tp
            df_metrics.acc[i]=acc
            df_metrics.prec[i]=prec
            df_metrics.recall[i]=recall
            df_metrics.F1_score[i]=F1_score
            df_metrics.kappa_cohen[i]=kappa_cohen
            df_metrics.auc[i]=auc


        display(df_metrics)
        eval(btag).append(df_metrics)


# In[19]:


l=[all0,all1,all2,orig,real]

with open("rf_trainvar_semvar.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)
    
with open("rf_trainvar_semvar.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


# In[20]:


with open("rf_trainvar_semvar.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


# In[21]:


dfall0=pd.DataFrame()
dfall1=pd.DataFrame()
dfall2=pd.DataFrame()
orig=pd.DataFrame()
real=pd.DataFrame()

ldf=[dfall0,dfall1,dfall2,orig,real]

ldfmean=ldf.copy()
ldfstd=ldf.copy()



for j in range (5):
    ldf[j]=b[j][0]
    for i in range(len(b[j])):
        if i!=0:
            ldf[j] = pd.concat([ldf[j], b[j][i]]).apply(pd.to_numeric)
            
    ldfmean[j]=ldf[j].groupby(level=0).mean() 
    ldfstd[j]=ldf[j].groupby(level=0).std()        


# In[22]:


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(["500","750","1000","1250","1500"], ldfmean[0].F1_score, label="0 jets b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[1].F1_score, label="1 jet b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[2].F1_score, label="2 jets b")
ax.plot(["500","750","1000","1250","1500"], ldfmean[3].F1_score, label="Proporción jets b: 86%-13%-1%")
ax.plot(["500","750","1000","1250","1500"], ldfmean[4].F1_score, label="Proporción jets b: 65%-33%-2%")

ax.fill_between(["500","750","1000","1250","1500"], ldfmean[0].F1_score - ldfstd[0].F1_score, ldfmean[0].F1_score + ldfstd[0].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[1].F1_score - ldfstd[1].F1_score, ldfmean[1].F1_score + ldfstd[1].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[2].F1_score - ldfstd[2].F1_score, ldfmean[2].F1_score + ldfstd[2].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[3].F1_score - ldfstd[3].F1_score, ldfmean[3].F1_score + ldfstd[3].F1_score, alpha=0.3)
ax.fill_between(["500","750","1000","1250","1500"], ldfmean[4].F1_score - ldfstd[4].F1_score, ldfmean[4].F1_score + ldfstd[4].F1_score, alpha=0.3)



ax.set_xlabel('Masa de test', fontsize=11)
ax.set_ylabel('F1-score', fontsize=11)
ax.legend(fontsize='large',loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle('Comparación de F1-score para las diferentes configuraciones de b-tag' , fontsize=14)
plt.show()

