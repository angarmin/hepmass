#!/usr/bin/env python
# coding: utf-8

# **Libraries**

# In[ ]:


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

cwd ="/home/angela/Notebook/machine_learning/normalizados/RF_bucle/"
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore') #ATENCION QUE ESTO CREO QUE FUNCIONA PARA TODO EL NOTEBOOK

import random
random.seed(6)
np.random.seed(6)
np.random.RandomState(6)

from sklearn.preprocessing import MinMaxScaler
import pickle

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


# **LOAD**

# In[92]:


#ALL THE DATASET

df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainpickle")
df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testpickle")


# In[93]:


#SIMPLe DATASET
#df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainsimplepickle")#.sample(n=1000, random_state=1)
#df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testsimplepickle")#.sample(n=300, random_state=1)


# # Machine learning

# **We will follow the scikit-learn schema to train and validate the model**
# 
# 
# https://scikit-learn.org/stable/modules/cross_validation.html
# 

# ## RF 

# In[94]:


#Scores


def Scores(y_true,y_pred):
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    prec=tp / (tp + fp)
    recall= tp / (tp + fn)
    F1_score= 2 * (prec * recall) / (prec + recall)
    acc=metrics.accuracy_score(y_true,y_pred)
    kappa_cohen=metrics.cohen_kappa_score(y_true,y_pred)
        
    return(tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen)


def model(X_train,y_train,X_test,rs):
    
    scaler = MinMaxScaler()
    scaler=scaler.fit(X_train)
    
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    model = RandomForestClassifier()
    params = {'n_estimators':[350],'min_samples_leaf':[1],'max_features':['sqrt'],'criterion':['entropy'], 'random_state':[rs]}
 
    
    grid = GridSearchCV(estimator=model, param_grid=params,cv=5,verbose=1, n_jobs=-1)

    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_
    
    best_model.fit(X_train,y_train)
    # Predict test set labels
    y_pred = best_model.predict(X_test)    
    y_pred_proba = best_model.predict_proba(X_test)[::,1] #Neccesary to make the ROC curve 

    return(grid,best_model,y_pred,y_pred_proba)


# In[95]:


df.columns


# In[ ]:


#Creación lista F1 score
#Creación lista importancias
F1Scoreslist=[]
Importanciaslist=[]

run=1

for rs in range(5):  #rs será el random state, para ir cambianod de trains y semillas 
    #primero creo los dos df que necesito: scores e importancias
    
    df_score = pd.DataFrame(index=[500,750,1000,1250,1500], columns=[list(range(27,0,-1))])
    
    df_import= pd.DataFrame(columns=['lep_pt', 'lep_eta', 'lep_phi', 'met_miss', 'met_phi',
       'jets_no', 'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_btag', 'jet2_pt',
       'jet2_eta', 'jet2_phi', 'jet2_btag', 'jet3_pt', 'jet3_eta', 'jet3_phi',
       'jet3_btag', 'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_btag', 'm_jj',
       'm_jjj', 'm_lv', 'm_jlv', 'm_wwbb'])
    
    np.random.seed(rs)
    
#separo por masas: 

    for i in (500,750,1000,1250,1500):

        print('mass=', i )

        dfmass1000=df.loc[df['mass'] ==i]        
        dfmass0=df.loc[df['mass'] == 0].sample(random_state=rs,n=dfmass1000.shape[0])   #cojo todos!!!!!!!!!!!!!! 
        dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(random_state=rs,frac=1).reset_index(drop=True)    #concatenating and shuffling
        dfmass1000['mass']=dfmass1000['mass'].astype(float)
        dfmass1000=dfmass1000.sample(random_state=rs,n=20000*5)
        #print("train \n", pd.crosstab(dfmass1000['label'],dfmass1000['mass']))
        dfmass1000=dfmass1000.drop('mass', axis=1)        #la masa es una caracterísitca


        #El test tambien lo cambio, porque no estoy comparando con ningún modelo
        dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
        dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(random_state=rs,n=dfmass1000test.shape[0])
        dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
        dfmass1000test=dfmass1000test.sample(random_state=rs,n=10000*5)

        #print("test \n", pd.crosstab(dfmass1000test['label'],dfmass1000test['mass']))  

        dfmass1000test=dfmass1000test.drop('mass', axis=1)  



       # print(X_train)
       # print(y_test)
        
        importancias=["label",'lep_pt', 'lep_eta', 'lep_phi', 'met_miss', 'met_phi',
       'jets_no', 'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_btag', 'jet2_pt',
       'jet2_eta', 'jet2_phi', 'jet2_btag', 'jet3_pt', 'jet3_eta', 'jet3_phi',
       'jet3_btag', 'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_btag', 'm_jj',
       'm_jjj', 'm_lv', 'm_jlv', 'm_wwbb']        
        
        for j in range(27,0,-1): #características
            
            dfmass1000=dfmass1000[importancias]
            dfmass1000test=dfmass1000test[importancias]
            
            
            X_train=dfmass1000.drop(['label'], axis=1)
            y_train=dfmass1000.label

            X_test=dfmass1000test.drop(['label'], axis=1)
            y_test=dfmass1000test.label
            
            print(run)
            run=run+1

            grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test,rs)
            
            
            d= dict(zip(X_train.columns, best_model.feature_importances_))
            d=pd.DataFrame(data=d, index=[i])
            
            #append the row: 
            if j==27:
                df_import = df_import.append(d,ignore_index=False)
                                                  

        #METRICS 

            tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen=Scores(y_test,y_pred)

            
            #meto la puntuación al dataset 
            df_score.loc[i ,j ] = F1_score

            #print("X_train_shape",X_train.shape)
            #print("X_test_shape",X_test.shape)
            
            
            
            #Elimino una característica: 
            d = {'Valor': best_model.feature_importances_, 'Nombres': X_train.columns}
            importancias = pd.DataFrame(data=d)
            importancias=importancias.sort_values(by=['Valor'], ascending=False)
            importancias.Nombres= importancias.Nombres.astype('category')
            importancias.Valor= importancias.Valor.astype('float')
            importancias= ["label"]+list(importancias.Nombres[0:-1])
            

        
        
        F1Scoreslist.append(df_score)
        Importanciaslist.append(df_import)
            


# In[ ]:


with open("F1Scoreslist.txt", "wb") as fp:   #Pickling
    pickle.dump(F1Scoreslist, fp)

with open("Importanciaslist.txt", "wb") as fp:   #Pickling
    pickle.dump(Importanciaslist, fp)


# In[ ]:


with open("F1Scoreslist.txt", "rb") as fp:   # Unpickling
    F1Scoreslist = pickle.load(fp)
    
with open("Importanciaslist.txt", "rb") as fp:   # Unpickling
    Importanciaslist = pickle.load(fp)


# ### Gráfica F1-Score

# In[ ]:


df_mean = pd.concat(F1Scoreslist)
df_mean=df_mean.astype(float)
df_mean=df_mean.groupby(df_mean.index).mean() 

df_std= pd.concat(F1Scoreslist)
df_std=df_std.astype(float)
df_std=df_std.groupby(df_std.index).std() 


# In[ ]:


fig, ax = plt.subplots(figsize=(12,6))
ax.plot(range(27,0,-1),df_mean.loc[500], label="mass= 500")
ax.plot(range(27,0,-1),df_mean.loc[750], label="mass= 750")
ax.plot(range(27,0,-1),df_mean.loc[1000], label="mass= 1000")
ax.plot(range(27,0,-1),df_mean.loc[1250], label="mass= 1250")
ax.plot(range(27,0,-1),df_mean.loc[1500], label="mass= 1500")


ax.fill_between(range(27,0,-1), df_mean.loc[500] - df_std.loc[500], df_mean.loc[500] + df_std.loc[500], alpha=0.3)
ax.fill_between(range(27,0,-1), df_mean.loc[750] - df_std.loc[750], df_mean.loc[750] + df_std.loc[750], alpha=0.3)
ax.fill_between(range(27,0,-1), df_mean.loc[1000] - df_std.loc[1000], df_mean.loc[1000] + df_std.loc[1000], alpha=0.3)
ax.fill_between(range(27,0,-1), df_mean.loc[1250] - df_std.loc[1250], df_mean.loc[1250] + df_std.loc[1250], alpha=0.3)
ax.fill_between(range(27,0,-1), df_mean.loc[1500] - df_std.loc[1500], df_mean.loc[1500] + df_std.loc[1500], alpha=0.3)



ax.set_xlabel('Número de variables', fontsize=13)
ax.set_ylabel('F1-score', fontsize=13)
ax.legend(fontsize='large',loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle('Comparación de F1-score para las diferentes configuraciones de b-tag' , fontsize=14)
plt.show()


# ### Gráficas características

# In[ ]:


df_mean = pd.concat(Importanciaslist)
df_mean=df_mean.astype(float)
df_mean=df_mean.groupby(df_mean.index).mean() 

df_std= pd.concat(F1Scoreslist)
df_std=df_std.astype(float)
df_std=df_std.groupby(df_std.index).std() 


# In[ ]:


for i in (500,750,1000,1250,1500):

    df_mean=df_mean.iloc[:, np.argsort(df_mean.loc[i])]
    df_std=df_std.iloc[:, np.argsort(df_mean.loc[i])]

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(5,9))
    ax.barh(df_mean.columns, df_mean.loc[i], xerr=df_std.loc[i], align='center')
    y_pos = np.arange(len(df_mean.columns))
    ax.set_xlabel('Masa = '+str(i), fontsize=13 )
    ax.set_title('Ranking de características Random Forest', fontsize=16)
    plt.show()
    plt.clf()


# In[ ]:




