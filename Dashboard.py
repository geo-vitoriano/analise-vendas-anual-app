import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
warnings.filterwarnings('ignore') # ou warnings.filterwarnings(action='once')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn import metrics
from http.client import IncompleteRead

st.set_page_config(layout = 'wide')
#st.title("Análise Anual")

# Paleta de cores ################################################################################################################
AZUL1, AZUL2, AZUL3, AZUL4, AZUL5 = '#03045e', '#0077b6', "#00b4d8", '#90e0ef', '#CDDBF3'
CINZA1, CINZA2, CINZA3, CINZA4, CINZA5 = '#212529', '#495057', '#adb5bd', '#dee2e6', '#f8f9fa'
VERMELHO1, LARANJA1, AMARELO1, VERDE1, VERDE2 = '#e76f51', '#f4a261',	'#e9c46a', '#4c956c', '#2a9d8f'

# Extração de dados ####################################################################################################################
lista_arquivos = []
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2020.zip')
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2021.zip')
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2022.zip')
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2023.zip')

dataframes = []

for arquivo in lista_arquivos :
  df = pd.read_csv(arquivo, sep=';', compression='zip')
  dataframes.append(df)

dados = pd.concat(dataframes, ignore_index=True)
dados.sort_values(by=['Emissão Certificado'], inplace=True)

# Tratamento de dados #################################################################################################################

# Reduzindo o dataset para os campos que interessam na modelagem.
dados = dados[['Emissão Certificado', 'Desconto', 'Valor', 'Qtd', 'Atacado', '1o. Agrupamento']]
#dados = dados[['Emissão Certificado', 'Desconto', 'Valor', 'Qtd', 'Atacado']]

# Renomeando os campos
dados = dados.rename(columns={"Emissão Certificado": "data",
                              "Valor": "valor",
                              "Qtd": "qtd",
                              '1o. Agrupamento':'uf',
                              "Atacado" : "atacado",
                              "Desconto": "desconto"})
# Alterando os tipos de dados
dados['data'] = pd.to_datetime(dados['data'], errors='coerce')
dados['valor'] = dados['valor'].str.replace(',','.').astype(float)
dados['desconto'] = dados['desconto'].astype(str)
dados['desconto'] = dados['desconto'].str.replace(',','.').astype(float)

# Alterando a informação do campo para binária.
for index, row in dados.iterrows() :
  if row['atacado'] == 'Sim' :
     dados.at[index, 'atacado'] = 1
  else :
    dados.at[index, 'atacado'] = 0

# Gerando a totalização por dia e atacado.
#dados = dados.groupby(['data', 'atacado'])[['valor', 'desconto', 'qtd']].sum()
#dados = dados.reset_index()
    
dados_timeline = dados.groupby(['data'])[['valor', 'desconto', 'qtd']].sum()
dados_timeline = dados_timeline.reset_index()    

# Criação de novos campos para a modelagem
dados_timeline['dia_do_ano'] = dados_timeline['data'].dt.dayofyear
dados_timeline['mes'] = dados_timeline['data'].dt.month
dados_timeline['dia_semana'] = dados_timeline['data'].dt.dayofweek
dados_timeline['dia_do_mes'] = dados_timeline['data'].dt.day
dados_timeline['mes_ano'] = dados_timeline['data'].dt.month.map(str) + '/' + dados_timeline['data'].dt.year.map(str)
dados_timeline['ano'] = dados_timeline['data'].dt.year


# Retirando os outliars
filtro = dados_timeline['valor'] > 50.00
dados_timeline = dados_timeline[filtro]

df_vendas_ano = dados_timeline.groupby(['ano'])[['valor']].sum()
df_vendas_ano = df_vendas_ano.reset_index()

# Gráfico por ano ###################################################################################################
fig_vendas_ano = px.bar(df_vendas_ano,
                        x='ano',
                        y='valor',
                        text_auto = True,
                        title = 'Análise Anual')

#st.dataframe(dados)

# Gráfico mensal por Mês ano #########################################################################################

receita_mensal = dados_timeline.set_index('data').groupby(pd.Grouper(freq='M'))['valor'].sum().reset_index()
receita_mensal['ano'] = receita_mensal['data'].dt.year
receita_mensal['mes'] = receita_mensal['data'].dt.month_name()
#receita_mensal['valor'] = receita_mensal['valor'] / 1000

fig_receita_mensal = px.line(receita_mensal,
                     x = 'mes',
                     y = 'valor',
                     markers = True,
                     range_y = (0, receita_mensal.max()),
                     color='ano',
                     line_dash = 'ano',
                     title = 'Receita mensal')

fig_receita_mensal.update_layout(yaxis_title = 'Receita')

# Gráfico de top 7 de estados com maior receita ######################################################################################

df_vendas_uf_ano = dados.groupby('uf')[['valor']].sum().reset_index().sort_values(by=['valor'], ascending=False)
df_vendas_uf_ano = df_vendas_uf_ano[df_vendas_uf_ano['uf'] != '0']
#df_vendas_uf_ano['valor'] = np.log2(df_vendas_uf_ano['valor'])
top_uf = df_vendas_uf_ano[:7] 
top_uf = top_uf.reset_index(drop=True)
#top_uf.sort_values(by=['valor'], ascending=True, inplace=True)

# Definindo a paleta de cores
AZUL1, AZUL2, AZUL3, AZUL4, AZUL5 = '#03045e', '#0077b6', "#00b4d8", '#90e0ef', '#CDDBF3'
CINZA1, CINZA2, CINZA3, CINZA4, CINZA5 = '#212529', '#495057', '#adb5bd', '#dee2e6', '#f8f9fa'
VERMELHO1, LARANJA1, AMARELO1, VERDE1, VERDE2 = '#e76f51', '#f4a261',	'#e9c46a', '#4c956c', '#2a9d8f'

fig_vendas_uf = px.bar(top_uf,
                        x='valor',
                        y='uf',
                        text_auto = True,
                        color = 'uf',
                        title = 'Análise por estado')

# Dashboad ###########################################################################################################################
coluna_1, coluna_2 = st.columns(2)

with coluna_1 :
  st.plotly_chart(fig_vendas_ano, use_container_width=True)
with coluna_2 :
  #st.plotly_chart(fig_receita_mensal, use_container_width=True)  
  st.plotly_chart(fig_vendas_uf, use_container_width=True)

st.plotly_chart(fig_receita_mensal, use_container_width=True) 