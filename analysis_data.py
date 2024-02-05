#%%
import collections

from utils import *


#%%
# DF_LEITURAS (stage 1)
df_leituras = pd.read_feather('Data/_feathers/df_leituras')

#%%
#####################
# Checking for NaNs #
'''doesn't have'''
df_leituras.isnull().sum()

# %%
del df_leituras

#%%
# DF_LEITURAS (Stage 2) #
df_leituras = pd.read_feather('Data/_feathers/df_leituras_stage2')

readings_2021 = df_leituras[(df_leituras['Data/Hora'] >= '2021-01-01 00:00:00') 
                          & (df_leituras['Data/Hora'] <= '2022-01-01 00:00:00')]
del df_leituras

#%%
#####################################
# MISSING VALUES (15 min registers) #

"""
Todos os locais começam a '2021-01-01';
2 locais terminam antes de '2022-01-01 00:00:00'.

Local   :: Data do último registo
1148958 -> 2021-08-31 11:00:00
853712  -> 2021-11-12 08:00:00

DROP de ambos os locais
"""
local_lst = readings_2021.Local.unique()
data = []

for l in tqdm(local_lst):
    df_a = readings_2021[readings_2021.Local == l]['Data/Hora']
    min = df_a.min()
    max = df_a.max()
    data.append([l, min, max])
        
df_obs = pd.DataFrame(data, columns=['Local', 'min', 'max'])
df_obs.sort_values(by=['max', 'min'])

#%%
#####################################
# MISSING VALUES (1H registers) #

'''
COLUNA MISSING
    * == 0 valor real
    * > 0 valor em falta
        == 1 (não tratado)
        == 2 ("removido"/descartado)
    * < 0 valor imputado
        == -1 (média, interpolação, valor ant/post, ..)
        == -2 ("feature engineering para obter valor real)
'''
#%%
################################################################################
#  REG_H  #  STAGE 1  #
################################################################################

# IMPORT REG_H | STAGE 1 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_1')

#%%
'''
 0.0    4225412
 1.0       4644
-2.0       1507

Nota: ainda sem a remoção dos locais com +50 missing values
'''
reg_H.missing.value_counts()

#%%
# Zero changes
zero_changes = reg_H[reg_H.missing == -2].index
print(len(zero_changes))  # 1507

#%%
# 233 (eram 220?!) registos na condição "zero_changes" para o Local 1092537
'''
1092537	233
880973	55
1217984	22
1083252	21
1191802	21
'''
reg_H.loc[zero_changes].groupby(by=['Local'])   \
                       .count()                 \
                       .sort_values(by=['Leitura'], ascending=False)

#%%
# nº de registos (Hora) em falta
n_miss = reg_H[(reg_H.missing == 1)].groupby(by=['Local']).count()
aux1 = n_miss.sort_values(by=['Data/Hora'], ascending=False)
aux1 = aux1[['Data/Hora']]
aux1['Percent'] = (aux1['Data/Hora']/8761)*100        # 8761 = (24*365) + 1

print("TOTAL = " + str(aux1['Data/Hora'].sum()))

"""
Local -> no. registos (Hora) em falta [desses, +1H em falta para ambos os lados]
915653 -> 1035 [1030]
868426 -> 287 [279]
1192000 -> 279 [276]
[1092537 -> 220 [216]
1144200 -> 152 [149]
2288907 -> 118 [115]
978574 -> 115 [114]
_________________________
996343 -> 48 [46]
997102 -> 45 [43]
...

"""
aux1.head(20)
#%%
'''
miss_ant/pos == 1 => existe reg 30 min antes/dps
miss_ant/pos == 2 => existe reg 45 min antes/dps
miss_ant/pos == 3 => existe reg 60 min antes/dps

Ex.: 

      3    2    1    0   reg   0    1    2    3
    __|____.____.____.____|____.____.____.____|__
     00   15   30   45   00   15   30   45   00

'''
# 1H (ou mais) em falta para ambos os lados
plus1H_miss = reg_H[(reg_H.missing == 1) & (reg_H.miss_ant >= 3) & (reg_H.miss_pos >= 3)]
aux2 = plus1H_miss.groupby(by=['Local']).count().sort_values(by=['Data/Hora'], ascending=False)
aux2 = aux2[['Data/Hora']]
aux2['Percent'] = (aux2['Data/Hora']/8761)*100        # 8761 = (24*365) + 1

print("TOTAL = " + str(aux2['Data/Hora'].sum()))
aux2.head(20)

# CONCLUSÃO: Drop dos locais com mais de 50 registos (ou 1% dos registos) em falta
# Removidos 6 Locais. Restam 477 após remoção. 

#%%
reg_H.missing.value_counts()

#%%
################################################################################
#  REG_H  #  STAGE 2  #
################################################################################

# IMPORT REG_H | STAGE 2 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_2')

# %%
# Registos não tratados e "removidos"/descartados [missing > 0]
regs = reg_H[reg_H.missing > 0].groupby(by=['Local'])                           \
                               .count()[['Data/Hora']]                          \
                               .sort_values(by=['Data/Hora'], ascending=False)
regs.rename(columns = {'Data/Hora': 'n_regs'}, inplace = True)
n_locals = reg_H.Local.nunique()
print(f'"Removidos"/em falta {regs.n_regs.sum()} registos (Horas), de {n_locals*8761} possíveis, dos {n_locals} locais restantes')

regs
#%%
################################################################################
#  REG_H  #  STAGE 3  #
################################################################################

# IMPORT REG_H | STAGE 3 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_3')

#%%
# Notas: 
#   * Todos os registos com missing > 0 têm a Leitura a NaN
#   * Removidos registos da primeira hora do ano seguinte (2022) 
#     '2022-01-01 00:00:00' (477, um por Local)
'''
Missing
-2.0       1493
-1.0        420
---------------
 0.0    4174369
 2.0       2076
 1.0        162

'''
reg_H.missing.value_counts()

#%%
'''
Todos os registos com missing > 0 (2238 registos) com leitura a NaN
'''
reg_H[reg_H.missing > 0].Leitura.value_counts(dropna=False)

#%%
# Nota: Todos os registos com missing > 0 têm a Leitura a NaN
# e consequentemente Consumo a 'NaN' também.
reg_H[reg_H.missing > 0].Consumo.unique()

#%%
################################################################################
# CONSUMOS NEGATIVOS
################################################################################

reg_H[reg_H.Consumo < 0].sort_values(by='Consumo')[['Data/Hora',
                                                    'Hora_ant',
                                                    'Leitura',
                                                    'Local', 
                                                    'Consumo']]

#%%
reg_H[(reg_H.Local == 1212842) & (reg_H.Month == 9) & (reg_H.Day == 23)]


# %%
# DF_LEITURAS (Stage 2) #
df_leituras = pd.read_feather('Data/_feathers/df_leituras_stage2')

df_leituras
# %%
pd.set_option('display.max_columns', None)

df_cd = pd.merge(reg_H[reg_H.Consumo < 0], 
                 df_leituras[['Data/Hora', 'Local', 'Leitura', 'Consumo']],
                 how='inner',
                 left_on=['Hora_ant', 'Local'],
                 right_on=['Data/Hora', 'Local'],
                 suffixes=('_reg_H', '_df_leituras'))

df_cd

#%%
# Observação: Todos os registos com Consumo < 0 têm missing == 0
df_cd.missing.value_counts()

#%%
df_cd[['Local',
       'Data/Hora_reg_H', 
       'Hora_ant', 
       'Data/Hora_df_leituras', 
       'Leitura_reg_H',
       'Leitura_df_leituras',
       'Consumo_reg_H',
       ]]

#%%
df_cd


# %%
df_leituras.info()
# %%
df_leituras[(df_leituras.Local == 1212842) 
          & (df_leituras['Data/Hora'] == '2021-09-23 15:00:00')]
# %%
reg_H[(reg_H.Local == 1212842) 
    & (reg_H['Data/Hora'] == '2021-09-23 15:00:00')]

# %%
df_leituras[(df_leituras.Local == 1212842) 
          & (df_leituras['Data/Hora'] == '2021-09-23 14:00:00')]
# %%
reg_H[(reg_H.Local == 1212842) 
    & (reg_H['Data/Hora'] == '2021-09-23 14:00:00')]

# %%
df_leituras[(df_leituras.Local == 1212842) 
          & (df_leituras['Data/Hora'] == '2021-09-23 16:00:00')]

# %%
reg_H[(reg_H.Local == 1212842) 
    & (reg_H['Data/Hora'] == '2021-09-23 16:00:00')]
# %%
'''
HOUVE UM PROBLEMA COM:
* O CÁLCULO DO CONSUMO  
'''
################################
#%%

#%%
# 51 locais com consumo negativo (1713)
negativos_count = reg_H[reg_H.Consumo < 0].groupby(by=['Local', 'Month']).count()[['Leitura']].sort_values(by='Leitura', ascending=False)
dfi.export(negativos_count, 'imgs/df_negativos_count.png')

# ocorrência de consumos negativos por Local e mês
negativos_count
#%%
negativos = reg_H[reg_H.Consumo < 0].sort_values(by=['Local', 'Data/Hora'])[['Local', 'Data/Hora', 'Leitura', 'Consumo']]
negativos.to_csv('negativos.csv', index=False)

negativos
#%%
# Locais com consumo negativo (51)
neg_locals = reg_H[reg_H.Consumo < 0].Local.unique()
len(neg_locals)

#%%
# Registos com consumo negativo (1713)
len(reg_H[reg_H.Consumo < 0])

#%%
'''
51 locais  1713 registos

@Local     @Consumos negativos 
1046349    641
1203533    606
854719     157
1202715    128
905097      41
819450      36
1030256     30
833312      10
2050692      9
906816       7
'''
reg_H[reg_H.Consumo < 0].groupby(by=['Local']).count().sort_values(by='Leitura', ascending=False)[['Data/Hora']].head(16)

#%%
################################################################################
# CONSUMOS EXCESSIVOS
################################################################################

# Consumo superiores ao possível para o respectivo calibre
cadastro = pd.read_feather('Data/_feathers/df_cadastro')
cadastro = cadastro[cadastro.Local.isin(reg_H.Local.unique())]

cadastro

#%%
'''nesta seleção de locais apenas tem contadores com calibre 15'''
cadastro.Calibre.value_counts()

#%%
# coluna Caudal Max
reg_H = reg_H.join(cadastro.set_index('Local'), on='Local')
reg_H['Caudal Max'] *= 1000

reg_H

#%%
# Contagem de registos de consumo que ultrapassam o caudal máx (por Local)
reg_H[reg_H.Consumo > reg_H['Caudal Max']].groupby(by='Local').count().sort_values(by='Leitura', ascending=False)[['Leitura']]
#%%
# Registos que ultrapassam o caudal máx (ordenado por ordem decrescente de Consumo)
# Caudal Max = 3125 em todos
reg_H[reg_H.Consumo > reg_H['Caudal Max']].sort_values(by='Consumo', ascending=False)[['Data/Hora', 'Local', 'Consumo', 'Leitura']]

#%%
# Locais com consumo excessivo (95)
over_locals = reg_H[reg_H.Consumo > reg_H['Caudal Max']].Local.unique()
len(over_locals)

#%%
# Registos com consumo excessivo (99)
excessos = reg_H[reg_H.Consumo > reg_H['Caudal Max']][['Data/Hora', 'Local', 'Leitura', 'Consumo']]

# ordenados por Local seguido de Data/Hora
df_excessos_ord_Local = excessos.sort_values(by=['Local', 'Data/Hora'])
dfi.export(df_excessos_ord_Local, 'imgs/df_excessos_ord_Local.png')

# ordenados por Consumo (ordem decrescente)
df_excessos_ord_Consumo = excessos.sort_values(by='Consumo', ascending=False)
dfi.export(df_excessos_ord_Consumo, 'imgs/df_excessos_ord_Consumo.png')

df_excessos_ord_Consumo

#%%
# Registos com consumo excessivo (Distribuição temporal)
reg_H[reg_H.Consumo > reg_H['Caudal Max']].groupby(by=['Month', 'Day']).count()[['Leitura']]

#%%
# Locais com consumo negativo ou excessivo = 134
'''1812 registos'''
len(set(list(neg_locals) + list(over_locals)))

#%%
locals_to_del = set(list(neg_locals) + list(over_locals))
len(locals_to_del)

#%%
neg_exc_locals = reg_H[(reg_H.Consumo < 0) | (reg_H.Consumo > reg_H['Caudal Max'])].Local.unique()
len(neg_exc_locals)

#%%
collections.Counter(list(locals_to_del)) == collections.Counter(list(neg_exc_locals))

#%%
reg_H.Local.nunique()

#%%
################################################################################
# GERAÇÃO DE GRÁFICOS #
################################################################################
'''
@Local     @Consumos negativos 
1046349    641
1203533    606
854719     157
1202715    128
905097      41
819450      36
1030256     30
833312      10
2050692      9
906816       7
'''
   
plot_leitura_consumo(reg_H, 1046349)

#%%
plot_leitura_consumo(reg_H, 1203533)

#%%
plot_leitura_consumo(reg_H, 854719)

#%%
plot_leitura_consumo(reg_H, 1202715)


#%%
plot_leitura_consumo(reg_H, 905097)

#%%
plot_leitura_consumo(reg_H, 819450)

#%%
plot_leitura_consumo(reg_H, 1030256)

#%%
plot_leitura_consumo(reg_H, 833312)

#%%
plot_leitura_consumo(reg_H, 2050692)

#%%
plot_leitura_consumo(reg_H, 906816)

################################################################################
################################################################################

# IMPORT REG_H | STAGE 4 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_4')

#%%
reg_H.Local.nunique()

#%%
reg_H[reg_H.Consumo < 0]

#%%
################################################################################
# ANÁLISE
##########
#%%
consumo_describe = reg_H.groupby(by='Local')['Consumo'].describe()
consumo_describe

#%%
consumo_describe.sort_values(by='std')

#%%
consumo_describe.sort_values(by='max')

#%%
consumo_describe.sort_values(by='mean')

#%%
leitura_describe = reg_H.groupby(by='Local')['Leitura'].describe()
leitura_describe.sort_values(by=['std'], ascending=True)
###############################################################################


#%%
# Registo de Consumos Zero consecutivos

locals_lst = reg_H.Local.unique()
consec_dict = {}

for l in locals_lst:
    consumos = reg_H[reg_H.Local == l]['Consumo'].to_numpy()
    consec = []
    count = 0
    #print(consumos)
    for cons in consumos:
        #print(cons)
        if cons == 0:
            count += 1
        else:
            count = 0
        #print(count)
        consec.append(count)

    consec_dict[l] = consec
    print(f'Local {l} -> {len(consec)}')  # 8760 registos num ano

reg_H['Zero_consec'] = 0
for l in locals_lst:
    reg_H.loc[reg_H.Local == l, 'Zero_consec'] = consec_dict[l]

reg_H['Zero_consec'] = reg_H['Zero_consec'].astype('int64')


#%%
'''
168 Horas = 7 dias
8760 Horas = 365 dias
'''
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(reg_H[['Local', 'Zero_consec']].groupby(by=['Local']).max().sort_values(by='Zero_consec', ascending=False))

#%%
df = reg_H[['Local', 'Zero_consec']].groupby(by=['Local']).max().sort_values(by='Zero_consec', ascending=False)
df.query("Zero_consec >= 168")  # 124 Locais com pelo menos 7 dias consecutivos sem consumo

#%%
# Drop do Local 831069 que apenas tem consumo 0
reg_H.drop(reg_H[reg_H.Local == 831069].index, inplace=True)

#%%
reg_H.info()

#%%
reg_H.Local.nunique()



#%%
from utils import *


# IMPORT REG_H | STAGE 5 #
# Já removidos Locais com consumos excessivos, negativos 
# e sem qualquer consumo registado
# 342 Locais restantes
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_5')

reg_H
# %%
reg_H.info()
# %%
# 342 Locais
reg_H.Local.nunique()

#%%
'''
Data/Hora                   0
Local                       0
Leitura                  1787
miss_ant              2992585
miss_pos              2992585
Hora_ant                    0
Hora_pos              2992585
Leitura_ant           2992585
Leitura_pos           2992585
Leitura_diff          2992585
Consumo                  2062
consec_H                    0
consec_D                 3335
Day                         0
Month                       0
Year                        0
Hour                        0
Minute                      0
missing                     0
consec_miss_reg          3335
Calibre                     0
Caudal Max                  0
Tipo de Instalação          0
'''
reg_H.isnull().sum()

#%%
'''
COLUNA MISSING
    * == 0 valor real
    * > 0 valor em falta
        == 1 (não tratado)
        == 2 ("removido"/descartado)
    * < 0 valor imputado
        == -1 (média, interpolação, valor ant/post, ..)
        == -2 ("feature engineering para obter valor real)
'''
# 2062 consumo.isnull()
# 1787 missing > 0
reg_H[(reg_H.Consumo.isnull()) & (reg_H.missing > 0)]

#%%
reg_H[reg_H.missing == 1]

#%%
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#%%
# Registos do Local l, e Dia d do mês m
l = 826782
m = 2
d = 27
reg_H[(reg_H.Local == l) & (reg_H.Month == m) & (reg_H.Day == d)]         \
    [['Local', 'Data/Hora', 'Leitura', 'Consumo', 'missing']]

#%%
################################################################################
'''
COLUNA MISSING
    * == 0 valor real
    * > 0 valor em falta
        == 1 (não tratado)
        == 2 ("removido"/descartado)
    * < 0 valor imputado
        == -1 (média, interpolação, valor ant/post, ..)
        == -2 ("feature engineering para obter valor real)

Contagem:
-2.0       1203
-1.0        345
 0.0    2992585
 1.0        126
 2.0       1661

 (2.0 e 1.0) com valores em falta
 1661 + 126 = 1787 Leituras por preencher
'''
reg_H.missing.value_counts()

#%%
reg_H[reg_H.Leitura.isnull()]

#%%
# Dataframe com todos os missing values (1787 registos)
missing_values = reg_H[reg_H.missing > 0]
missing_values

#%%
#[ADD coluna "year_day" e "week_day"]
missing_values['year_day'] = missing_values['Data/Hora'].apply(lambda x: x.timetuple().tm_yday)
missing_values['week_day'] = missing_values['Data/Hora'].apply(lambda x: (x.dayofweek + 1) % 7) # para manter dom=0, seg=1, ..
#%%
missing_values.groupby(by=['miss_ant']).count()[["Data/Hora"]]

#%%
missing_values.groupby(by=['miss_pos']).count()[["Data/Hora"]]

#%%
# Igual distância temporal entre a última e próxima leitura
# 33 registos
missing_values[missing_values.miss_ant == missing_values.miss_pos].sort_values(by='miss_ant')
#%%

# missing_values.Local.nunique()
# reg_H.Local.nunique()

# %%
# Nº de missing values por Local 
# [196 Locais com missing values (de 342)
#  +/-57.3% dos Locais]
'''
Local   no. missing values
996343	46
942464	32
1034162	31
1223836	28
1087673	28
1163922	27
839361	23
1108980	22
1066374	21
1074113	21
1212745	21
'''
missing_values.groupby(by=['Local'])                                  \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Data/Hora'], ascending=False)

#%%
# Maiores gaps em relação a registos anteriores
'''
Data/Hora           Local   Leitura  miss_ant   miss_pos
2021-01-29 07:00:00	996343	NaN	     134.0      2.0
2021-01-16 18:00:00	942464	NaN	     117.0	    4.0
2021-06-12 04:00:00	1034162	NaN	     95.0   	2.0
2021-06-12 04:00:00	1223836	NaN	     91.0	    4.0
2021-06-12 01:00:00	1074113	NaN	     85.0	    3.0
2021-05-13 19:00:00	1212745	NaN	     65.0	    4.0
'''

missing_values.sort_values(by=['miss_ant'], ascending=False)

#%%
# Maiores gaps em relação a registos posteriores
'''
Data/Hora           Local   Leitura  miss_ant  miss_pos  Hora_ant             Hora_pos
2021-01-27 22:00:00	996343	NaN	     2.0	   134.0	 2021-01-27 21:15:00  2021-01-29 07:45:00
2021-01-15 14:00:00	942464	NaN	     5.0	   116.0	 2021-01-15 12:30:00  2021-01-16 19:15:00
2021-06-11 05:00:00	1034162	NaN	     3.0	   94.0      2021-06-11 04:00:00  2021-06-12 04:45:00
2021-06-11 06:00:00	1223836	NaN      3.0	   92.0	     2021-06-11 05:00:00  2021-06-12 05:15:00
2021-06-11 05:00:00	1074113	NaN	     5.0	   83.0	     2021-06-11 03:30:00  2021-06-12 02:00:00
'''
missing_values.sort_values(by=['miss_pos'], ascending=False)

# %%
# Nº de missing values por mês
# 8 Meses com missing values
'''
Month
2	  978
6	  485
11	  134
1	  118
8	  40
5	  18
3     7
9	  7
'''
missing_values.groupby(by=['Month'])                                  \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Data/Hora'], ascending=False)

#%%
# Nº de missing values em cada dia (ordenado por nº de missing values)
# 24 dias com missing values
'''
Month  Day	
2	   27	877
6	   11	415
11	    1	134
2	   12	67
6	   12	52
8	   10	35
1	   15	30
       28	28
       16	27
2	   2	26
'''
missing_values.groupby(by=['Month', 'Day'])                           \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Data/Hora'], ascending=False)

#%%
# Mês | Dia | Local | nº de missing values desse Local naquele dia
missing_values.groupby(by=['Month', 'Day', 'Local'])                            \
              .count()[['Data/Hora']]                                           \
              .sort_values(by=['Month', 'Day', 'Data/Hora'], ascending=False)

#%%
# Nº de missing values em cada dia (ordenado por mês e dia)
missing_values.groupby(by=['Month', 'Day', 'Local'])                  \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Month', 'Day', 'Data/Hora'], ascending=False)   \
              .groupby(by=['Month', 'Day']).sum()

#%%
# Mês | Dia  | nº de locais com missing values nesse dia
missing_values.groupby(by=['Month', 'Day', 'Local'])                  \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Month', 'Day', 'Data/Hora'], ascending=False)   \
              .groupby(by=['Month', 'Day']).count()


#%%
# Contagem de nº de horas em falta para cada dia, de cada local 
missing_values.groupby(by=['Local', 'Month', 'Day'])                  \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Data/Hora'], ascending=False)
              #.sort_values(by=['Local', 'Month', 'Day'], ascending=False)

#%%
'''
missing_values.groupby(by=['Local', 'Month', 'Day'])                  \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Month', 'Day'], ascending=False)
'''
#%%
# apresenta mês, dia e horas em falta em cada um desses dias
missing_values.groupby(by=['Local', 'Month', 'Day', 'Hour'])          \
              .count()[['Data/Hora']]                                 \
              .sort_values(by=['Local', 'Month', 'Day'], ascending=False)

#%%
# 227 Horas (Mes, dia e Hora exatos) com missing values nas 24*365 = 8760 possíveis (2.6%)
# Contagem [sem discriminar o Local]
missing_values.groupby(by=['Month', 'Day', 'Hour'])                             \
              .count()[['Data/Hora']]                                           \
              .sort_values(by=['Month', 'Day'], ascending=False)

#%%
missing_values.groupby(by=['year_day', 'Month', 'Day'])                         \
              .count()[['Data/Hora']]

#%%
# dom 0, seg 1, ter 2, qua 3, qui 4, sex 5, sab 6
missing_values.groupby(by=['week_day'])                                         \
              .count()[['Data/Hora']]

#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values.groupby(by=['Local', 'Month', 'Day'])                  \
                        .count()[['Data/Hora']]                                 \
                        .sort_values(by=['Local', 'Data/Hora'], ascending=False))


#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values.groupby(by=['Hour'])                                   \
                        .count()[['Data/Hora']]                                 \
                        .sort_values(by=['Data/Hora'], ascending=False))
    # %%
reg_H[(reg_H.missing < 0) & (reg_H.Month == 2)]
