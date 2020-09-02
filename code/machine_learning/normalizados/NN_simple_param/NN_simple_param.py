
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


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import os 
cwd = os.getcwd() + "/"
cwd="/home/angela/Notebook/machine_learning/normalizados/NN_simple_param/"
from sklearn import metrics

from sklearn.neural_network import MLPClassifier


warnings.filterwarnings('ignore') #ATENCION QUE ESTO CREO QUE FUNCIONA PARA TODO EL NOTEBOOK


# **LOAD**

# In[2]:


#ALL THE DATASET

df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainpickle")
df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testpickle")


# In[3]:


#SIMPLe DATASET
#df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainsimplepickle")
#df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testsimplepickle")


# # Machine learning

# **We will follow the scikit-learn schema to train and validate the model**
# 
# 
# https://scikit-learn.org/stable/modules/cross_validation.html
# 

# ## SVM

# In[4]:


#Scores

def Scores(y_true,y_pred):
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn+fp)
    sens=tp/(tp+fn)
    acc=metrics.accuracy_score(y_true,y_pred)
    kappa_cohen=metrics.cohen_kappa_score(y_true,y_pred)
        
    return(sens,spec,acc,kappa_cohen,tn, fp, fn, tp)

def model(X_train,y_train,X_test):
    
    model = MLPClassifier() 
    
    #grid search, copy of kaggle: https://www.kaggle.com/hatone/mlpclassifier-with-gridsearchcv#L72
    #defalult (scikit learn) :  best_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)))

    #params = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}    #model = SVC(probability=True) 
    #reduced: 
    params = {'solver': ['lbfgs'], 'max_iter': [200,600,1000,1400,1800], 'alpha': 10.0 ** -np.arange(4, 6), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0]}    #model = SVC(probability=True) 

    grid = GridSearchCV(estimator=model, param_grid=params,cv=5,verbose=1)

    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_
    
    best_model.fit(X_train,y_train)
    # Predict test set labels
    y_pred = best_model.predict(X_test)    
    y_pred_proba = best_model.predict_proba(X_test)[::,1] #Neccesary to make the ROC curve 

    return(grid,best_model,y_pred,y_pred_proba)



# ## Entrenando y testeando con 1000

# In[5]:


cwd1=cwd+"con1000"


# In[6]:


#Metric df: 
df_metrics = pd.DataFrame(index=[500,750,1000,1250,1500], columns=['accuracy','sensitibity','specifity','kappa','auc'])


for i in (500,750,1000,1250,1500):

    print('mass=', i )

    #train
    dfmass0=df.loc[df['mass'] == 0]     #0 same size of dfmass1000 #TENER EN CUENTA QUE AQUI QUE LA DE 0 SEA MAYOR SIZE ES CASUALIDAD
    #EN OTRO PICKE PODRIA SER AL REVES, ESTO SE DEBERIA DE ARREGLAR
    dfmass0['mass'] = np.random.choice([500,750,1000,1250,1500], dfmass0.shape[0])#las cambio para queno sean 0 porque entonces sabría el label
    dfmass1000=df.loc[df['mass'] !=0].sample(n=dfmass0.shape[0], random_state=1,replace=True)                                                      #cojo todos!!!!!!!!!!!!!! 
    dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(frac=1).reset_index(drop=True)    #concatenating and shuffling
    dfmass1000['mass']=dfmass1000['mass'].astype(float)
    dfmass1000=dfmass1000.sample(n=28000*5, random_state=1)         #la masa es una caracterísitca
    
    print("train", pd.crosstab(dfmass1000['label'],dfmass1000['mass']))
    
    #test
    dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
    dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(n=dfmass1000test.shape[0], random_state=1)
    dfmass0test['mass'] = np.random.choice([i], dfmass0test.shape[0])#las cambio para queno sean 0 porque entonces sabría el label
    dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
    dfmass1000test['mass']=dfmass1000test['mass'].astype(float)
    dfmass1000test=dfmass1000test.sample(n=14000*5, random_state=1)
    
    print("test", pd.crosstab(dfmass1000test['label'],dfmass1000test['mass']))

    X_train=dfmass1000.drop(['label'], axis=1)
    y_train=dfmass1000.label
    
    X_test=dfmass1000test.drop(['label'], axis=1)
    y_test=dfmass1000test.label
        
   # print(X_train)
   # print(y_test)

    grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test)
    print(best_model)

    sens,spec,acc,kappa,tn, fp, fn, tp=Scores(y_test,y_pred)
    print('accuracy', acc)
    print('sensitibity', sens)
    print('specifity', spec)
    print('kappa', kappa)
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)


    #Roc curve construction

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig(cwd1+'MASS=' + str(i) + 'ROC.png')
    plt.show()
    plt.clf() 
    print('AUC', auc)
    
   
    df_metrics.accuracy[i]=acc
    df_metrics.sensitibity[i]=sens
    df_metrics.specifity[i]=spec
    df_metrics.kappa[i]=kappa
    df_metrics.auc[i]=auc
    print(X_train.shape)
    print(X_test.shape)
    



# In[7]:


print(X_train.shape)
print(X_test.shape)
y_test.value_counts()


# In[8]:


fig, ax = plt.subplots()
ax.plot(["500","750","1000","1250","1500"], df_metrics["accuracy"], label="accuracy")
ax.plot(["500","750","1000","1250","1500"], df_metrics["sensitibity"], label="sensitibity")
ax.plot(["500","750","1000","1250","1500"], df_metrics["specifity"], label="specifity")
ax.plot(["500","750","1000","1250","1500"], df_metrics["auc"], label="auc")
ax.plot(["500","750","1000","1250","1500"], df_metrics["kappa"], label="kappa")

ax.set_xlabel('mass')
ax.set_ylabel('Metric')
legend = ax.legend(fontsize='x-large')
plt.show()
plt.savefig(cwd1+ 'metrics_comparation.png')


# In[9]:


cwd2=cwd+"sin1000"


# In[10]:


#Metric df: 
df_metrics = pd.DataFrame(index=[500,750,1000,1250,1500], columns=['accuracy','sensitibity','specifity','kappa','auc'])


for i in (500,750,1000,1250,1500):

    print('mass=', i )

    #train

    dfmass1000=df.loc[(df['mass'] !=0) & (df['mass'] !=1000)] 
    dfmass0=df.loc[df['mass'] == 0].sample(n=dfmass1000.shape[0], random_state=1,replace=True)    #0 same size of dfmass1000
    dfmass0['mass'] = np.random.choice([500,750,1250,1500], dfmass0.shape[0])#las cambio para queno sean 0 porque entonces sabría el label
    dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(frac=1).reset_index(drop=True)    #concatenating and shuffling
    dfmass1000['mass']=dfmass1000['mass'].astype(float)
    dfmass1000=dfmass1000.sample(n=28000*5, random_state=1)         #la masa es una caracterísitca
    print("train", pd.crosstab(dfmass1000['label'],dfmass1000['mass']))
    
    #test
    dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
    dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(n=dfmass1000test.shape[0], random_state=1)
    dfmass0test['mass'] = np.random.choice([i], dfmass0test.shape[0])#las cambio para queno sean 0 porque entonces sabría el label
    dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
    dfmass1000test['mass']=dfmass1000test['mass'].astype(float)
    dfmass1000test=dfmass1000test.sample(n=14000*5, random_state=1)         #la masa es una caracterísitca
    print("train", pd.crosstab(dfmass1000test['label'],dfmass1000test['mass']))
    
    X_train=dfmass1000.drop(['label'], axis=1)
    y_train=dfmass1000.label
    
    X_test=dfmass1000test.drop(['label'], axis=1)
    y_test=dfmass1000test.label
        
   # print(X_train)
   # print(y_test)

    grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test)
    print(best_model)

    sens,spec,acc,kappa,tn, fp, fn, tp=Scores(y_test,y_pred)
    print('accuracy', acc)
    print('sensitibity', sens)
    print('specifity', spec)
    print('kappa', kappa)
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)


    #Roc curve construction

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig(cwd2+'MASS=' + str(i) + 'ROC.png')
    plt.show()
    plt.clf() 
    print('AUC', auc)
    
   
    df_metrics.accuracy[i]=acc
    df_metrics.sensitibity[i]=sens
    df_metrics.specifity[i]=spec
    df_metrics.kappa[i]=kappa
    df_metrics.auc[i]=auc
    
    print(X_train.shape)
    print(X_test.shape)
    


# In[11]:


print(X_train.shape)
print(X_test.shape)
y_test.value_counts()


# In[12]:


fig, ax = plt.subplots()
ax.plot(["500","750","1000","1250","1500"], df_metrics["accuracy"], label="accuracy")
ax.plot(["500","750","1000","1250","1500"], df_metrics["sensitibity"], label="sensitibity")
ax.plot(["500","750","1000","1250","1500"], df_metrics["specifity"], label="specifity")
ax.plot(["500","750","1000","1250","1500"], df_metrics["auc"], label="auc")
ax.plot(["500","750","1000","1250","1500"], df_metrics["kappa"], label="kappa")

ax.set_xlabel('mass')
ax.set_ylabel('Metric')
legend = ax.legend(fontsize='x-large')
plt.show()
plt.savefig(cwd2+ 'metrics_comparation.png')

