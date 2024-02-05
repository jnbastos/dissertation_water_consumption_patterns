# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:56:02 2021

@author: elisaaraujo
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

#from preprocess import import_dataset

def import_dataset(directory_path, delimitador, encoding = None):
    df = pd.DataFrame()
    for file_name in glob.glob(directory_path+'*.csv'):
        x = pd.read_csv(file_name, sep=delimitador, engine='python', encoding=encoding)
        x['filename'] = file_name
        df = pd.concat([df, x], axis=0, sort=False)
        df = df.reset_index(drop=True)
    return df

#%%
''' Ler o dataset'''
directory_path = 'Data/Cadastro/'
df_cadastro = import_dataset(directory_path,'[;]', encoding='utf-8')

list_locals_cadastro=list(df_cadastro.Local.unique())


#%%
df_cadastro

#%%
list_locals_cadastro
#%%
#remove duplicate rows
df_cadastro.sort_values('filename',inplace=True)
df_cadastro.drop_duplicates(subset=['MoradaCompleta','Local'], keep= 'last', inplace=True)
df_cadastro.sort_index(inplace=True)
df_cadastro.reset_index(drop=True, inplace=True)

#%%
df_cadastro['Tipo de Instalação'].value_counts(dropna=False)
#%%
cadastro_domestico = df_cadastro[df_cadastro['Tipo de Instalação'] == '1 DOMÉSTICO']
tipo_nan = df_cadastro[(df_cadastro['Tipo de Instalação'] == '?') | (df_cadastro['Tipo de Instalação'].isna())]

#%%
df_cadastro.Calibre.value_counts(dropna=False)

#%%
"""
Calibre -> Caudal máximo (m^3/h)
DN 15 3.125
DN 20 5
DN 25 7.5875
DN 30 12.5
DN 32 12.5
DN 40 20
DN 50 32.25
DN 65 50
"""
caudal = {15: 3.125,
          20: 5,
          25: 7.5875,
          30: 12.5,
          32: 12.5,
          40: 20,
          50: 32.25,
          65: 50,
          }

df_cadastro = df_cadastro[df_cadastro.Calibre.isin(list(caudal.keys()))]
df_cadastro

#%%
df_cadastro.drop_duplicates(subset=['Local'], keep='last', inplace=True)
df_cadastro['Caudal Max'] = df_cadastro['Calibre'].replace(caudal)
#%%
def ref_non_digit(obj):
    if str(obj).isnumeric():
        return int(obj)
    else: 
        return -1

df_cadastro['Local'] = df_cadastro.Local.apply(lambda x: ref_non_digit(x))
df_cadastro['Calibre'] = df_cadastro.Calibre.apply(lambda x: ref_non_digit(x))
df_cadastro = df_cadastro[(df_cadastro.Local != -1)
                        & (df_cadastro.Calibre != -1)]
df_cadastro = df_cadastro[['Local', 'Calibre', 'Caudal Max', 'Tipo de Instalação']]

df_cadastro.astype({'Local': int, 'Calibre': int})

#%%
df_cadastro.Calibre.value_counts()

#%%
df_cadastro.info()
#%%
#export dataset df_cadastro

#df_cadastro['Caudal Max'] = df_cadastro['Caudal Max'].astype(str)
df_cadastro.reset_index(drop=True, inplace=True)
df_cadastro.to_feather('Data/_feathers/df_cadastro')

# %%
