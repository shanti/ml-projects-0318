#!/usr/bin/env python
# coding: utf-8

# In[100]:


# Pacotes necessários para a análise exploratória
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# In[101]:


# Função para plotagem dos mapas de correlação
def plot_heatmap(height, data):
   plt.figure(figsize=(height,height*0.75))
   return sns.heatmap(data.corr(), vmin=-1, center=0, vmax=1)


# In[103]:


try:
    my_abs_path = Path("data.csv").resolve(strict=True)
except FileNotFoundError:
    data=pd.read_csv('http://www.aneel.gov.br/dados/relatorios?p_p_id=dadosabertos_WAR_dadosabertosportlet&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=gerarGeracaoFonteCSV&p_p_cacheability=cacheLevelPage&p_p_col_id=column-2&p_p_col_count=1')
    data.to_csv('data.csv')
else:
    data=pd.read_csv('data.csv')


# In[104]:


data.head()


# In[105]:


data['competencia'] = data['anoReferencia'] + data['mesReferencia'] / 12 - 1/12
data.head()


# In[108]:


data[data['nomFonteGeracao'] == 'Biomassas'].sort_values(by=['competencia'])


# In[109]:


data[data['nomFonteGeracao'] == 'Biomassas'].head()


# In[110]:


# Imprime a matriz de correlação entre as variáveis
ax = plot_heatmap(10, data)


# In[111]:


print(set(data['nomFonteGeracao']))


# In[112]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
data['FonteGeracao'] = label_encoder.fit_transform(data['nomFonteGeracao'])
pd.get_dummies(data['FonteGeracao']).rename(index=str, columns={0: "a", "1": "c"}).tail()


# In[115]:


data[['nomFonteGeracao','FonteGeracao']].tail(10)


# In[116]:


data.tail()


# In[66]:


# Imprime a matriz de correlação entre as variáveis - depois das transformações
ax = plot_heatmap(10, data)


# In[67]:


data[data['nomFonteGeracao'] == 0].sort_values(by=['competencia'])


# In[120]:


filtr = data[(data['FonteGeracao'] == 5) & (data['anoReferencia'] > 2010)]
plt.scatter(filtr['competencia'], filtr['mdaEnergiaDespachadaGWh'])


# In[118]:


filtr = data[(data['FonteGeracao'] == 6) & (data['anoReferencia'] > 2010)]
plt.scatter(filtr['competencia'], filtr['mdaEnergiaDespachadaGWh'])

