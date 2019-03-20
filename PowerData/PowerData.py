#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pacotes necessários para a análise exploratória
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path


# In[2]:


# Função para plotagem dos mapas de correlação
def plot_heatmap(height, data):
   plt.figure(figsize=(height,height*0.75))
   return sns.heatmap(data.corr(), vmin=-1, center=0, vmax=1)


# In[3]:


try:
    data_path = os.path.join(os.path.dirname(os.path.realpath('__file__')),'data.csv')
    my_abs_path = Path(data_path).resolve(strict=True)
except FileNotFoundError:
    data=pd.read_csv('http://www.aneel.gov.br/dados/relatorios?p_p_id=dadosabertos_WAR_dadosabertosportlet&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=gerarGeracaoFonteCSV&p_p_cacheability=cacheLevelPage&p_p_col_id=column-2&p_p_col_count=1')
    data.to_csv(data_path)
else:
    data=pd.read_csv(data_path)


# In[4]:


data.head()


# In[5]:


data['competencia'] = data['anoReferencia'] + data['mesReferencia'] / 12 - 1/12
data.head()


# In[6]:


data[data['nomFonteGeracao'] == 'Biomassas'].sort_values(by=['competencia']).head()


# In[7]:


# Imprime a matriz de correlação entre as variáveis
ax = plot_heatmap(10, data)


# In[8]:


print(set(data['nomFonteGeracao']))


# In[9]:


#data = data.sort_values(by=['competencia'])
#data = data.reset_index(drop=True)
#data.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
data['FonteGeracao'] = label_encoder.fit_transform(data['nomFonteGeracao'])
data_onehot = pd.get_dummies(data['FonteGeracao']).rename(index=str, columns={0: "is_biomass", 1: "is_coal", 2: "is_outSIN", 3:"is_eolic", 4:"is_natGas", 5:"is_hidrelectric", 6:"is_itaipu", 7:"is_nuclear", 8:"is_oil", 9:"is_residual"})
data_onehot.head()


# In[11]:


data_onehot.head()


# In[12]:


data.reset_index(drop=True, inplace=True)
data_onehot.reset_index(drop=True, inplace=True)
data = pd.concat([data, data_onehot], axis=1)
data.head()


# In[13]:


data = data.sort_values(by=['competencia'])
data.reset_index(drop=True, inplace=True)
data.head()


# In[14]:


legend = data[['nomFonteGeracao','FonteGeracao']].tail(10).sort_values(by=['FonteGeracao'])
legend = legend.reset_index(drop=True)
legend


# In[15]:


data = data.drop(columns=['Unnamed: 0','ideGeracaoFonte','nomFonteGeracao','dthProcessamento'])
data.head()


# In[35]:


plt.figure(figsize=(15,10))
for index, row in legend.iterrows():
    if index not in [5,9,8,1,2,7]:
        plt.plot('competencia','mdaEnergiaDespachadaGWh',data=data[(data['FonteGeracao'] == index) & (data['anoReferencia'] > 2002)], label=str(index) + ' - ' + row.nomFonteGeracao)
plt.legend()


# In[59]:


data['mdaEnergiaDespachadaGWh'] = data['mdaEnergiaDespachadaGWh'].fillna(0)


# In[18]:


# Imprime a matriz de correlação entre as variáveis
ax = plot_heatmap(10, data)


# In[60]:


t = data[['competencia','mdaEnergiaDespachadaGWh']][(data['FonteGeracao'] == 0)]
t.head()


# In[61]:


t['mdaEnergiaDespachadaGWh + 1'] = t['mdaEnergiaDespachadaGWh'].shift(-1)
t.head()


# In[83]:


train = t[t['competencia'] < 2016]
test = t[t['competencia'] >= 2016]
x_train = train['mdaEnergiaDespachadaGWh']
y_train = train['mdaEnergiaDespachadaGWh + 1']
x_test = test['mdaEnergiaDespachadaGWh']
y_test = test['mdaEnergiaDespachadaGWh + 1']


# In[76]:


x_train.head()


# In[79]:


y_train.head()


# In[110]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
x_train = x_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
lrModel = lr.fit(x_train, y_train)


# In[112]:


import math
print('R²: ',lrModel.score(x_train, y_train))
print('Corr.: ',math.sqrt(lrModel.score(x_train, y_train)))


# In[113]:


x_test.values.reshape(-1, 1).shape
y_pred = lrModel.predict(x_test.values.reshape(-1, 1))


# In[115]:


plt.plot(test['competencia'],test['mdaEnergiaDespachadaGWh'])
plt.plot(test['competencia'],y_pred)

