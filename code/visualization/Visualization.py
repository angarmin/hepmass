
# coding: utf-8

# **HEPMASS DATASET**
# https://archive.ics.uci.edu/ml/datasets/HEPMASS

# **Libraries**

# In[39]:


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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



from sklearn import metrics


warnings.filterwarnings('ignore') #ATENCION QUE ESTO CREO QUE FUNCIONA PARA TODO EL NOTEBOOK

pwdnorm= "/home/angela/Notebook/visualization/imagenes_normalizados/"
pwdsnorm="/home/angela/Notebook/visualization/imagenes_sin_norm/"
pwdest="/home/angela/Notebook/visualization/images/"


# # NORMALIZED DATA

# # Dataset description
#     

# Complete name of variables
# 
# • Momento transverso invariante del leptón
# • Pseudoradipity del leptón
# • Azimut del leptón
# • Momento faltante
# • Azimut_misp
# • Número de jets
# • Momento transverso del jet principal
# • Pseudorapidity del jet principal
# • Azimut del jet principal
# • Btagging del jet princial
# • Momento transverso del jet segundo
# • Pseudorapidity del jet segundo
# • Azimut del jet segundo
# • Btagging del jet segundo
# • Momento transverso del jet tercero
# • Pseudorapidity del jet tercero
# • Azimut del jet tercero
# • Btagging del jet tercero
# • Momento transverso del jet cuarto
# • Pseudorapidity del jet cuarto
# • Azimut del jet cuarto
# • Btagging del jet cuarto
# • Masa invariante de lnu
# • Masa invariante de jlnu
# • Masa invariante de jj
# • Masa invariante de jjj
# • Masa invariante de tt
# • Modelo
# • Masa

# In[40]:


#ALL THE DATASET

df=pd.read_pickle("../data/normalizados/trainpickle")
#df_originaltest=pd.read_pickle("../data/normalizados/testpickle")


# **I have read the test but I don't use after**

# In[66]:


#SIMPLe DATASET
#df=pd.read_pickle("../data/normalizados/trainsimplepickle")
#df_originaltest=pd.read_pickle("../data/normalizados/testsimplepickle")


# Dataset structure: 
# - All features are float, except output (binary), also the number of jets an b-tags
# - No missing data
# - Output is totally balanced (50-50)

# Proportion between train and test is 70 vs 35, we keep the test without explore to not make us previous asumptions. 
# 

# In[42]:


df.describe()


# # EDA 
# 

# ## Density plots

# The main objetive is to be able of classifying each mass group (BSM) in front of background (SM), we have fix mass of SM events to 0, just to keep in mind. 

# ### Features (binary), without knowing mass!!

# In[43]:


'''dfdensity=df.iloc[:,0:]
names=dfdensity.iloc[:,1:].columns

#Plot density: 

#UNCOMENT TO SEE THE DENSITY DIAGRAM:

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="label", ylim='own', figsize=(10,2))

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()    


#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="label", ylim='own', figsize=(10,6),hist="True", bins=50, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(i+'prueba.png')
'''



# ### 22 low-level features (mass)

# **Contingency table classes**

# In[44]:


df.mass.value_counts()
#mass are balanced


# In[45]:


#UNCOMENT TO SEE THE DENSITY DIAGRAM:

'''for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,2))

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()    
'''
dfdensity=df.iloc[:,1:]
names=dfdensity.iloc[:,:-1].columns

#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(pwdnorm +i + '.png')
    


# **REMOVING OUTLIERS**

# In[46]:


#Removing outliers to print diagrams: 

dfdensity=df.iloc[:,1:]
names=dfdensity.iloc[:,:-1].columns


#Plot histogram: 

for i in names: 
    if dfdensity[i].dtypes=='float64': 
        
        df_sinout = dfdensity.copy()
        
        if i == "m_jjj" or  i== "m_lv":

            df_sinout=df_sinout[~((df_sinout[i]-df_sinout[i].mean()).abs() > 0.5*df_sinout[i].std())]
            ### outliers are removing for each column and printed, for the next column the whole dataset is loaded again. 
            ### "normal column outliers > 3"" , m_jjj and m_ñv (high std) > 0.5, there is no criteria , is just the best way od visualize 
            print(i)
       
        else: 
            df_sinout=df_sinout[~((df_sinout[i]-df_sinout[i].mean()).abs() > 3*df_sinout[i].std())]


        plt.figure()
        fig, axes = joypy.joyplot(df_sinout, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                              grid=True, legend=False)

        # Decoration
        plt.title(i, fontsize=22)
        plt.show()
        fig.savefig(pwdnorm +i + '_outlier.png')


# ** DIFFERENCE BETWEEN BTAG CORRECT AND UNCORRECT** 

# In[47]:


dfdensity=df.iloc[:,1:].query('jet1_btag==1 and jet2_btag==1')
names=dfdensity.iloc[:,:-1].columns

#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(pwdnorm +i + '_btag_correct.png')    
    
dfdensity=df.iloc[:,1:].query('jet1_btag==0 or jet2_btag==0')

#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(pwdnorm +i + '_btag_uncorrect.png')    


# ## Correlation Plot

# In[67]:


plt.figure(figsize=(18,18))
corr_normaliz=df.corr()
g=sns.heatmap(corr_normaliz, mask=np.zeros_like(df.corr(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot = True, annot_kws = {"size" : 10})
plt.show()
fig = g.get_figure()
fig.savefig(pwdnorm  + 'correlation.png')    
#Keep in mind: categorical as number of jets are not HERE!



# High correlations between:  (>70)
#     - jet1_pt, jet2_pt
#     - jet2_pt, jet3_pt
#     - jet3_pt, jet4_pt
#     
#     - m_wwbb, jet1_pt
#     - m_wwbb, jet2_pt

# ## Manifold learning (TSNE)
# 
# Reduction to two variables

# In[49]:


'''tsne = manifold.TSNE(n_components=2, init='pca',n_iter=250)
X_tsne = tsne.fit_transform(df.drop('mass',axis=1))    #droping mass, obviously we cannot use
#np.unique(y)'''


# In[50]:


'''scatter_x = X_tsne[:,0]
scatter_y = X_tsne[:,1]
group = df.label
cdict = {0: 'green', 1: 'red'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g],label = g)
ax.legend()
plt.show()'''


# In[51]:


'''#grouping by mass: 
group = df.mass
cdict = {0: 'green', 500: 'blue',750: 'purple',1000: 'orange',1250: 'red' , 1500:'brown'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g],label = g)
ax.legend()
plt.show()
'''


# We cannot see practically anything. All points are overlaped, RESULT ARE SAVED IN A FILE

# ## Proportion of B-tag

# In[52]:


pd.crosstab([df.jet1_btag, df.jet2_btag], [df.jet3_btag, df.jet4_btag],
            rownames=['1_btag', "2_btag"],
            colnames=['3_btag', "4_btag"])


# Asumming btag greater corresponds to b quark type: (just taking  the small sample)   14000 DATA
# 
# - b,b,b,b = 0
# - b,b,b,0 = 0
# - b,b,0,b = 0
# - b,b,0,0 = 4405
# 
# - b,0,b,b = 0
# - b,0,b,0 = 2512
# - b,0,0,b = 1340
# - b,0,0,0 = 624
# 
# - 0,b,b,b = 0
# - 0,b,b,0 = 1101
# - 0,b,0,b = 1917
# - 0,b,0,0 = 513
# 
# - 0,0,b,b = 797
# - 0,0,b,0 = 417
# - 0,0,0,b = 300
# - 0,0,0,0 = 74
# 
# 
# 
# 

# **btag vs mass**

# In[53]:


#making crosstab

a=pd.crosstab([df.mass], [df.jet1_btag, df.jet2_btag, df.jet3_btag, df.jet4_btag],
            colnames=['1_btag', "2_btag",'3_btag', "4_btag"], rownames= ["mass"])
#correct names
a.columns=["0000","0001","0010","0011","0100","0101","0110","1000","1001","1010","1100"]
#adding mass
a = a.reset_index()
a["mass"] = a["mass"].astype('category')
#key-value
a=pd.melt(a, id_vars=['mass'], value_vars=a.columns[1:])
a.columns=['mass','btag','count']

plt.figure(figsize=(22,10))
g=sns.barplot(x="mass", y="count", hue="btag", data=a)

plt.show()
fig = g.get_figure()
fig.savefig(pwdsnorm  + 'btags.png')    


# # DATA WITHOUT NORMALIZATION (SAME CODE: )

# In[54]:


#load
df=pd.read_pickle("../data/sin_norm/trainpickle")
df_originaltest=pd.read_pickle("../data/sin_norm/testpickle")

#df=pd.read_pickle("../data/sin_norm/trainsimplepickle")
#df_originaltest=pd.read_pickle("../data/sin_norm/testsimplepickle")
df.describe()
df.mass.value_counts()


# ## plots

# In[55]:


#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(pwdsnorm +i + '.png')
    


# **REMOVING OUTLIERS**

# In[56]:


#Removing outliers to print diagrams: 

dfdensity=df.iloc[:,1:]
names=dfdensity.iloc[:,:-1].columns


#Plot histogram: 

for i in names: 
    if dfdensity[i].dtypes=='float64': 
        
        df_sinout = dfdensity.copy()
        
        if i == "m_jjj" or  i== "m_lv":

            df_sinout=df_sinout[~((df_sinout[i]-df_sinout[i].mean()).abs() > 0.5*df_sinout[i].std())]
            ### outliers are removing for each column and printed, for the next column the whole dataset is loaded again. 
            ### "normal column outliers > 3"" , m_jjj and m_ñv (high std) > 0.5, there is no criteria , is just the best way od visualize 
            print(i)
       
        else: 
            df_sinout=df_sinout[~((df_sinout[i]-df_sinout[i].mean()).abs() > 3*df_sinout[i].std())]


        plt.figure()
        fig, axes = joypy.joyplot(df_sinout, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                              grid=True, legend=False)

        # Decoration
        plt.title(i, fontsize=22)
        plt.show()
        fig.savefig(pwdsnorm +i + '_outlier.png')


# ** DIFFERENCE BETWEEN BTAG CORRECT AND UNCORRECT** 

# In[57]:


dfdensity=df.iloc[:,1:].query('jet1_btag==1 and jet2_btag==1')
names=dfdensity.iloc[:,:-1].columns

#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(pwdsnorm +i + '_btag_correct.png')    
    
dfdensity=df.iloc[:,1:].query('jet1_btag==0 or jet2_btag==0')

#Plot histogram: 

for i in names: 
    plt.figure()
    fig, axes = joypy.joyplot(dfdensity, column=[i], by="mass", ylim='own', figsize=(10,6),hist="True", bins=70, overlap=0,
                          grid=True, legend=False)

    # Decoration
    plt.title(i, fontsize=22)
    plt.show()
    fig.savefig(pwdsnorm +i + '_btag_uncorrect.png')    


# ## Correlation Plot

# In[65]:


plt.figure(figsize=(18,18))
corr_sin_norm=df.corr()
g=sns.heatmap(corr_normaliz, mask=np.zeros_like(df.corr(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot = True, annot_kws = {"size" : 10})
plt.show()
fig = g.get_figure()
fig.savefig(pwdsnorm  + 'correlation.png')    
#Keep in mind: categorical as number of jets are not HERE!



# ## Proportion of B-tag

# In[59]:


pd.crosstab([df.jet1_btag, df.jet2_btag], [df.jet3_btag, df.jet4_btag],
            rownames=['1_btag', "2_btag"],
            colnames=['3_btag', "4_btag"])


# **btag vs mass**

# In[60]:


#making crosstab

a=pd.crosstab([df.mass], [df.jet1_btag, df.jet2_btag, df.jet3_btag, df.jet4_btag],
            colnames=['1_btag', "2_btag",'3_btag', "4_btag"], rownames= ["mass"])
#correct names
a.columns=["0000","0001","0010","0011","0100","0101","0110","1000","1001","1010","1100"]
#adding mass
a = a.reset_index()
a["mass"] = a["mass"].astype('category')
#key-value
a=pd.melt(a, id_vars=['mass'], value_vars=a.columns[1:])
a.columns=['mass','btag','count']

plt.figure(figsize=(22,10))
g=sns.barplot(x="mass", y="count", hue="btag", data=a)

plt.show()
fig = g.get_figure()
fig.savefig(pwdsnorm  + 'btags.png')    


# # COMPARATIONS

# In[68]:


#DIFERENCE BETWEEN CORRELATION NORMALIZ AND WITHOUT

diff_corr=corr_sin_norm-corr_normaliz

plt.figure(figsize=(18,18))
g=sns.heatmap(diff_corr, mask=np.zeros_like(df.corr(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot = True, annot_kws = {"size" : 10})

plt.show()
fig = g.get_figure()
fig.savefig(pwdest  + 'correlation_difference.png')    
#Keep in mind: categorical as number of jets are not HERE!

