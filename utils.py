from time import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import gc # garbage collector
import sys
import re
import seaborn as sns
import dataframe_image as dfi

from tqdm import tqdm # taqaddum progress meter


def import_dataset(directory_path, delimitador, encoding = None):
    df = pd.DataFrame()
    dfs = []
    for file_name in tqdm(glob.glob(directory_path+'*.csv')):
        aux = pd.read_csv(file_name, sep=delimitador, engine='python', encoding=encoding)
        aux['filename'] = file_name
        dfs.append(aux)
    df = pd.concat(dfs, axis=0, sort=False)
    df = df.reset_index(drop=True)
    return df


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def get_mem_var():
    var_list = [(name, sys.getsizeof(value)) for name, value in globals().items()]
    for name, size in sorted(var_list, key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


sns.axes_style("whitegrid")
sns.set(style="ticks")
sns.set(rc={'figure.figsize': (15, 8)})
sns.despine()


def plot_leitura_consumo(reg_H, local):
    # LEITURA
    data = reg_H[(reg_H.Local == local)].copy()
    data['Cons_neg'] = 0
    data.loc[data.Consumo < 0, 'Cons_neg'] = 1
    
    if data.Cons_neg.nunique() == 2:
        palette=['tab:blue', 'tab:red']
    else:
        palette=['tab:blue']

    ax = sns.lineplot(data=data[['Data/Hora', 'Leitura', 'Cons_neg']], x="Data/Hora", y="Leitura", hue='Cons_neg', palette=palette, legend=False)
    ax.set_title('Local ' + str(local), fontsize=20)
    ax.set_xlabel('Data', fontsize=15)
    ax.set_ylabel('Leitura', fontsize=15)
    plt.savefig('imgs/'+ str(local) + '_leitura.png')
    plt.show()

    # CONSUMO

    '''
    ax = sns.lineplot(data=data[['Data/Hora', 'Consumo', 'Cons_neg']], x="Data/Hora", y="Consumo", hue='Cons_neg', palette=palette, legend=False)
    ax.set_title('Local ' + str(local), fontsize=20)
    ax.set_xlabel('Data', fontsize=15)
    ax.set_ylabel('Consumo', fontsize=15)
    '''
    data = data.dropna(subset=['Consumo'])
    data_hora_neg = data[data.Consumo < 0]['Data/Hora'].to_numpy()
    data_hora_pos = data[data.Consumo >= 0]['Data/Hora'].to_numpy() 

    #data_hora.shape = data_hora.shape
    consumo_neg = data[data.Consumo < 0]['Consumo'].to_numpy()
    consumo_neg.shape = data_hora_neg.shape
    consumo_pos = data[data.Consumo >= 0]['Consumo'].to_numpy()
    consumo_pos.shape = data_hora_pos.shape

    over_cons = data[data.Consumo > data['Caudal Max']]
    text = list(zip(over_cons['Data/Hora'], over_cons['Consumo']))

    if over_cons.shape[0] > 0:
        for i in range(len(text)):
            plt.text(
                    pd.to_datetime('2021-01-02'), 3500+(i*1800),
                    'Data: '+str(text[i][0])+' | Consumo: '+str(text[i][1]),
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
                    verticalalignment='bottom', horizontalalignment='left',
                    color='black', 
                    fontsize=13)

        plt.plot(data_hora_pos, consumo_pos, linestyle='-', color='tab:blue', linewidth=0.5)
        plt.fill_between(data_hora_pos, consumo_pos, color='tab:blue')
        plt.plot(data_hora_neg, consumo_neg, linestyle='-', color='tab:red', linewidth=0.5)
        plt.fill_between(data_hora_neg, consumo_neg, color='tab:red')
        plt.title('Local ' + str(local), fontsize=20)
        plt.xlabel('Data', fontsize=15)
        plt.ylabel('Consumo', fontsize=15) 
        
        plt.axhline(y=data['Caudal Max'].unique(), color='grey', linestyle='--', label='Consumo Máx. Ltrs/Hora (Calibre 15)')
        
        #plt.legend()
        plt.savefig('imgs/'+ str(local) + '_consumo_com_excesso.png')
        plt.show()

        data.drop(over_cons.index, inplace=True) 
    
    ax = sns.scatterplot(data=data[['Data/Hora', 'Consumo', 'Cons_neg']], x="Data/Hora", y="Consumo", hue='Cons_neg', palette=palette, legend=False, s=15)
    ax.set_title('Local ' + str(local), fontsize=20)
    ax.set_xlabel('Data', fontsize=15)
    ax.set_ylabel('Consumo', fontsize=15)
    plt.legend()
    plt.savefig('imgs/'+ str(local) + '_consumo_scatter.png')
    plt.show() 

    data_hora_neg = data[data.Consumo < 0]['Data/Hora'].to_numpy()
    data_hora_pos = data[data.Consumo >= 0]['Data/Hora'].to_numpy() 

    #data_hora.shape = data_hora.shape
    consumo_neg = data[data.Consumo < 0]['Consumo'].to_numpy()
    consumo_neg.shape = data_hora_neg.shape
    consumo_pos = data[data.Consumo >= 0]['Consumo'].to_numpy()
    consumo_pos.shape = data_hora_pos.shape

    #plt.plot(data_hora, consumo, linestyle='-', color='tab:blue', linewidth=0.5)
    plt.plot(data_hora_pos, consumo_pos, linestyle='-', color='tab:blue', linewidth=0.5)
    plt.fill_between(data_hora_pos, consumo_pos, color='tab:blue')
    plt.plot(data_hora_neg, consumo_neg, linestyle='-', color='tab:red', linewidth=0.5)
    plt.fill_between(data_hora_neg, consumo_neg, color='tab:red')
    plt.title('Local ' + str(local), fontsize=20)
    plt.xlabel('Data', fontsize=15)
    plt.ylabel('Consumo', fontsize=15)
    plt.savefig('imgs/'+ str(local) + '_consumo.png')
    plt.show()