#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Pacotes necessários para a análise exploratória
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[43]:


# Função para plotagem dos mapas de correlação
def plot_heatmap(height, data):
   plt.figure(figsize=(height,height*0.75))
   return sns.heatmap(data.corr(), vmin=-1, center=0, vmax=1)


# In[59]:


data=pd.read_csv('http://www.aneel.gov.br/dados/relatorios?p_p_id=dadosabertos_WAR_dadosabertosportlet&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=gerarGeracaoFonteCSV&p_p_cacheability=cacheLevelPage&p_p_col_id=column-2&p_p_col_count=1')


# In[60]:


data.head()


# In[69]:


data['competencia'] = data['anoReferencia'] + data['mesReferencia'] / 12 - 1/12
data.head()


# In[71]:


data[data['nomFonteGeracao'] == 0].sort_values(by=['competencia'])


# In[63]:


data[data['nomFonteGeracao'] == 'Biomassas'].head()


# In[64]:


# Imprime a matriz de correlação entre as variáveis
ax = plot_heatmap(10, data)


# In[65]:


from sklearn.preprocessing import LabelEncoder
#set(data['nomFonteGeracao'])
label_encoder = LabelEncoder()
#Agora faça a transformação dos dados com o método fit_transform(), veja:

data['nomFonteGeracao'] = label_encoder.fit_transform(data['nomFonteGeracao'])
#A variável valores_numericos recebe a lista de valores já codificados, veja:

data['nomFonteGeracao'].head()


# In[66]:


# Imprime a matriz de correlação entre as variáveis - depois das transformações
ax = plot_heatmap(10, data)


# In[67]:


data[data['nomFonteGeracao'] == 0].sort_values(by=['competencia'])


# In[77]:


filtr = data[(data['nomFonteGeracao'] == 0) & (data['anoReferencia'] > 2010)]
plt.scatter(filtr['competencia'], filtr['mdaEnergiaDespachadaGWh'])


# In[78]:


filtr = data[(data['nomFonteGeracao'] == 1) & (data['anoReferencia'] > 2010)]
plt.scatter(filtr['competencia'], filtr['mdaEnergiaDespachadaGWh'])


# In[81]:


filtr = data[(data['nomFonteGeracao'] == 2) & (data['anoReferencia'] > 2010)]
plt.scatter(filtr['competencia'], filtr['mdaEnergiaDespachadaGWh'])

