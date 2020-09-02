#!/usr/bin/env python
# coding: utf-8

# **Libraries**

# In[1]:


print(1)

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
#cwd = os.getcwd()
cwd="/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/"

from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import optimizers

#import tensorflow as tf
#from keras.models import Sequential 
#from keras.layers import  Dense, Flatten, Activation, Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras import optimizers


from IPython.display import display


import random
random.seed(1)
np.random.seed(1)
np.random.RandomState(1)

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore') #ATENCION QUE ESTO CREO QUE FUNCIONA PARA TODO EL NOTEBOOK

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(2)


# **LOAD**

# In[ ]:


#ALL THE DATASET

df=pd.read_pickle("/home/angela/Notebook/data/normalizados/trainpickle")
df_originaltest=pd.read_pickle("/home/angela/Notebook/data/normalizados/testpickle")

print("loaded")


# # Machine learning

# **We will follow the scikit-learn schema to train and validate the model**
# 
# 
# https://scikit-learn.org/stable/modules/cross_validation.html
# 

# ## KERAS

# In[ ]:


#Scores  

# Function to create model, required for KerasClassifier
def grid_model(optimizer='adam', activation='relu', neurons =5, hidden_layers=1,dropout=0.0):
    # create model
    model = Sequential()
    #first layer
    model.add(Dense(neurons,activation=activation, input_shape=(16,)))
    
    for i in range(hidden_layers):
        # Add one hidden layer
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout))

        
    #last layer: 
    model.add(Dense(1, activation='tanh'))
    optimizer= optimizers.Adam() #by default lr=0.001
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model  #Jordi dijo categorical cross entropy pero me da error. 




def Scores(y_true,y_pred):
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    prec=tp / (tp + fp)
    recall= tp / (tp + fn)
    F1_score= 2 * (prec * recall) / (prec + recall)
    acc=metrics.accuracy_score(y_true,y_pred)
    kappa_cohen=metrics.cohen_kappa_score(y_true,y_pred)
        
    return(tn, fp, fn, tp, acc, prec,recall, F1_score,kappa_cohen)

def model(X_train,y_train,X_test):
    
    scaler = MinMaxScaler()
    scaler=scaler.fit(X_train)    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = KerasClassifier(build_fn=grid_model, verbose=0)

        
    #batch_size = [10, 20, 50, 100]
    #epochs = [50, 100,150]
    #neurons= [10,50,100,150,200,300]
    #hidden_layers=[3,4,5,6]
    #dropout= [0,0.1,0.2,0.3]
    
    batch_size = [10, 25]
    epochs = [10,50]
    neurons= [10,100,150]
    hidden_layers=[3,4,5]
    dropout= [0,0.2]

    
    

    param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, hidden_layers= hidden_layers,dropout=dropout)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=5) #n_jobs=-1, solo 1 cpus

    # define the grid search parameters
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_
            
    y_pred_proba = best_model.model.predict_proba(X_test) #Neccesary to make the ROC curve 
    y_pred =  best_model.predict(X_test)

    return( grid.best_params_,best_model,y_pred,y_pred_proba)




# In[ ]:


#Metric df: 
df_metrics =  pd.DataFrame(index=[750,500], columns=["tn", "fp", "fn", "tp", "acc", "prec","recall","F1_score","kappa_cohen","auc"])

File_object =  open("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/a.txt","a") 
File_object.write("dddd")
File_object.close()

for i in (750,500):
    
    File_object =  open("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/a.txt","a") 
    File_object.write(str(i))
    File_object.close()

    print('mass=', i )

    #train
      #0 same size of dfmass1000 #TENER EN CUENTA QUE AQUI QUE LA DE 0 SEA MAYOR SIZE ES CASUALIDAD
    #EN OTRO PICKE PODRIA SER AL REVES, ESTO SE DEBERIA DE ARREGLAR
    np.random.seed(1)
    #dfmass0['mass'] = np.random.choice([500,750,1000,1250,1500], dfmass0.shape[0])#las cambio para queno sean 0 porque entonces sabría el label
    dfmass1000=df.loc[df['mass'] ==i]        
    dfmass0=df.loc[df['mass'] == 0].sample(random_state=1,n=dfmass1000.shape[0])   #cojo todos!!!!!!!!!!!!!! 
    dfmass1000=pd.concat([dfmass1000, dfmass0]).sample(random_state=1,frac=1).reset_index(drop=True)    #concatenating and shuffling
    dfmass1000['mass']=dfmass1000['mass'].astype(float)
    dfmass1000=dfmass1000.sample(random_state=1,n=20000*5)
    print("train", pd.crosstab(dfmass1000['label'],dfmass1000['mass']))
    dfmass1000=dfmass1000.drop('mass', axis=1)        #la masa es una caracterísitca
    
    
    #test
    dfmass1000test=df_originaltest.loc[df_originaltest['mass'] == i]
    dfmass0test=df_originaltest.loc[df_originaltest['mass'] == 0].sample(random_state=1,n=dfmass1000test.shape[0])
    dfmass1000test=pd.concat([dfmass1000test, dfmass0test]).sample(frac=1).reset_index(drop=True)
    dfmass1000test=dfmass1000test.sample(random_state=1,n=10000*5)

    print("test", pd.crosstab(dfmass1000test['label'],dfmass1000test['mass']))  
    
    dfmass1000test=dfmass1000test.drop('mass', axis=1)  
    
    

    X_train=dfmass1000.drop(['label'], axis=1)
    X_train=X_train.drop(['lep_phi','met_phi','jets_no','jet1_phi','jet1_btag','jet2_phi','jet2_btag','jet3_phi','jet3_btag','jet4_phi', 'jet4_btag'], axis=1)
    y_train=dfmass1000.label
    
    X_test=dfmass1000test.drop(['label'], axis=1)
    X_test=X_test.drop(['lep_phi','met_phi','jets_no','jet1_phi','jet1_btag','jet2_phi','jet2_btag','jet3_phi','jet3_btag','jet4_phi', 'jet4_btag'], axis=1)
    y_test=dfmass1000test.label
        
   # print(X_train)
   # print(y_test)
    
    #try:

    grid,best_model,y_pred,y_pred_proba = model(X_train,y_train,X_test)
    print(best_model)
    
    file2write=open("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/df"+ str(i)+ ".txt",'w')
    file2write.write(str(grid))
    file2write.close()


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
        
    #except:
       # print("No funcionó :(")
    df_metrics.to_pickle("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/df"+ str(i)+ ".pkl") 


print("X_train_shape",X_train.shape)
print("X_test_shape",X_test.shape)
df_metrics.to_pickle("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/df_metrics.pkl") 


#display(df_metrics)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
y_test.value_counts()


# In[ ]:


fig, ax = plt.subplots()
ax.plot(["500","750","1000","1250","1500"], df_metrics["acc"], label="accuracy")
ax.plot(["500","750","1000","1250","1500"], df_metrics["F1_score"], label="F1_score")
ax.plot(["500","750","1000","1250","1500"], df_metrics["kappa_cohen"], label="kappa")

ax.set_xlabel('mass')
ax.set_ylabel('Metric')
legend = ax.legend(fontsize='x-large')
plt.show()
plt.savefig(cwd+ 'metrics_comparation.png')


# In[ ]:


df_metrics.to_pickle("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/df_metrics.pkl") 


# In[2]:


df = pd.read_pickle("/home/angela/Notebook/machine_learning/normalizados/Neural_network_keras/df500.pkl") 


# In[3]:


df


# In[ ]:




