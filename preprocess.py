#%%
from utils import *
from datetime import timedelta

#%%
#Import dataset readings
dir_path = 'Data/Leituras/'
df_leituras = import_dataset(dir_path,'[;]', encoding='utf-8')

#%%
# Concat with previous readings (df_leituras)
df_leituras_old = pd.read_feather('Data/_feathers/df_leituras_old')
df_leituras = pd.concat([df_leituras_old, df_leituras], axis=0, sort=False)

#%%
# Keep only domestic instalations from previous study (485 Locals)
# removing meters with already known anomalies
df_medias_hora_old = pd.read_feather('Data/_feathers/df_medias_hora_6meses_old')

#%%
df_domestic = df_medias_hora_old[df_medias_hora_old['Tipo de Instalação'] == '1 DOMÉSTICO']
df_leituras = df_leituras[df_leituras.Local.isin(df_domestic.Local.unique())]
df_leituras.Local.nunique()

#%%
# clear memory
del df_leituras_old, df_domestic

# garbage collector
gc.collect()

# gc.isenabled()

#%%
# [ADD column "filename"]
# Show file name without the path
df_leituras['filename'] = df_leituras['filename'].apply(lambda x: re.compile('(AdNorte-).*.csv').search(x)[0])

# [DEL column "Módulo"]
df_leituras.drop(columns=['Módulo'], inplace=True)

# "Data/Hora" conversion to datetime and df_leituras ordination
df_leituras['Data/Hora'] = pd.to_datetime(df_leituras['Data/Hora'])
df_leituras.sort_values(by=['Local', 'Data/Hora'], inplace=True)

'''#%%
df_leituras

#%%
# df_leituras to csv
df_leituras.to_csv('Data/_csv/df_leituras_join.csv', index=False)
'''

#%%
'''
24.796.380 entries
Data/Hora | Local | Leitura | filename
'''
df_leituras.info()

#%%
# Duplicates removal [removed 2.070.410 duplicated entries]
df_leituras.sort_values(by=['Local', 'Data/Hora', 'filename'], inplace=True)
df_leituras.drop_duplicates(subset=['Data/Hora', 'Local'], keep='last', inplace=True)

# [ADD column "Consumo"]
# Consumption every 15 minutes
df_leituras['Consumo'] = df_leituras.groupby('Local')['Leitura'].diff()

# EXPORT DF_LEITURAS (stage 1) #
'''
22.725.970 entries
 #   Column     Dtype         
---  ------     -----         
 0   Data/Hora  datetime64[ns]
 1   Local      object        
 2   Leitura    float64       
 3   filename   object        
 4   Consumo    float64 
'''
df_leituras.reset_index(drop=True, inplace=True)
df_leituras.to_feather('Data/_feathers/df_leituras')

#%%
# MISSING READINGS #
# Expected one reading every 15 min

# min and max datetime of each Local
df_continuas = df_leituras.groupby('Local').agg(
    data_inicial=('Data/Hora', np.min), data_final=('Data/Hora', np.max))

# no. of readings for each Local
df_n_readings = df_leituras.groupby('Local')['Data/Hora'].count().to_frame()
df_n_readings.rename(columns={'Data/Hora':'n_registos'}, inplace=True)
df_n_readings.reset_index(inplace=True)

df_continuas = pd.merge(df_continuas, df_n_readings, on='Local')
del df_n_readings
gc.collect()

# number of missing (15 min) registers
df_continuas['n_dias'] = (df_continuas.data_final - df_continuas.data_inicial) / np.timedelta64(1, 'D')
df_continuas['reg_15min'] = df_continuas['n_dias'] * 4 * 24
df_continuas['diff_registos'] = df_continuas['reg_15min'] - df_continuas['n_registos']

# EXPORT DF_CONTINUAS #
# leituras_continuas = verificacao_leituras_continuas
'''
485 entries
 #   Column          Dtype         
---  ------          -----         
 0   Local             int64         
 1   data_inicial      datetime64[ns]
 2   data_final        datetime64[ns]
 3   n_registos        int64         
 4   n_dias            float64       
 5   reg_15min         float64       
 6   diff_registos     float64 
'''
df_continuas.to_feather('Data/_feathers/leituras_continuas')
      
#%%
# [ADD columns "Hora_ant", "consec_miss_reg", "consec_H"]
# "Hora_ant" - timestamp of the last registed reading
# "consec_miss_reg" - no. of consecutive missed readings (15 minutes periods) counting from the last reading
# "consec_H" - no. of hours since last reading

# A optimizar! Ocupa muita memória e tempo
df_dict = df_leituras.to_dict('records')
df_dict

first_local_row = True
for row in tqdm(df_dict):
    if first_local_row:
        hora_ant = row['Data/Hora']
        row['Hora_ant'] = hora_ant
        first_local_row = False
        data_final = df_continuas[df_continuas['Local'] == row['Local']]['data_final'].item()
    else:
        row['Hora_ant'] = hora_ant
        hora_ant = row['Data/Hora']
    
    row['Data/Hora']
    if (row['Data/Hora'] == df_continuas[df_continuas['Local'] == row['Local']]['data_final']).item():
        first_local_row = True

df_leituras = pd.DataFrame.from_dict(df_dict)
df_leituras['consec_miss_reg'] = ((df_leituras['Data/Hora'] - df_leituras['Hora_ant']) / np.timedelta64(15, 'm')) - 1
df_leituras['consec_H'] = df_leituras['consec_miss_reg']*0.25

#%%
# EXPORT DF_LEITURAS (Stage 2) #
# with info about missing readings
'''
22.725.970 entries

 #   Column           Dtype         
---  ------           -----         
 0   Data/Hora        datetime64[ns]
 1   Local            int64         
 2   Leitura          float64       
 3   filename         object        
 4   Consumo          float64       
 5   Hora_ant         datetime64[ns]
 6   consec_miss_reg  float64       
 7   consec_H         float64 
'''
df_leituras.to_feather('Data/_feathers/df_leituras_stage2')

#%%
#%%
from utils import *

#%%
# IMPORT DF_LEITURAS (Stage 2) #
df_leituras = pd.read_feather('Data/_feathers/df_leituras_stage2')

#%%
# Readings of year 2021 
# (from 2021-01-01 00:00:00 to 2022-01-01 00:00:00)
'''
16.953.754 entries
'''
readings_2021 = df_leituras[(df_leituras['Data/Hora'] >= '2021-01-01 00:00:00') 
                          & (df_leituras['Data/Hora'] <= '2022-01-01 00:00:00')]

del df_leituras

#%%
# [ADD columns "consec_D", "Month", "Day", "Hour", "Minute"]
readings_2021 = readings_2021.assign(consec_D=lambda x: x['consec_H'] / 24,
                                     Month=lambda x: x['Data/Hora'].dt.month,
                                     Day=lambda x: x['Data/Hora'].dt.day,
                                     Hour=lambda x: x['Data/Hora'].dt.hour,
                                     Minute=lambda x: x['Data/Hora'].dt.minute)

#%%
# Dropping Locals [1148958, 853712] for its registers 
# Locals with readings ending before '2022-01-01 00:00:00'.
# Last reading:
# 1148958 -> 2021-08-31 11:00:00
# 853712  -> 2021-11-12 08:00:00
# Droped 53516 rows (readings every 15 min)
drops_idx = readings_2021[(readings_2021.Local == 1148958) 
                        | (readings_2021.Local == 853712)].index
readings_2021.drop(drops_idx, inplace=True)

#%%
# EXPORT READINGS_2021 #
'''
16900238 entries

 #   Column           Dtype         
---  ------           -----         
 0   Data/Hora        datetime64[ns]
 1   Local            int64         
 2   Leitura          float64       
 3   filename         object        
 4   Consumo          float64       
 5   Hora_ant         datetime64[ns]
 6   consec_miss_reg  float64       
 7   consec_H         float64       
 8   consec_D         float64       
 9   Month            int64         
 10  Day              int64         
 11  Hour             int64         
 12  Minute           int64
 '''
readings_2021.reset_index(drop=True, inplace=True)
readings_2021.to_feather('Data/_feathers/readings_2021')

#%%
# IMPORT <= READINGS_2021 #
readings_2021 = pd.read_feather('Data/_feathers/readings_2021')
readings_2021

#%%
# ############
#    REG_H
# ############

# Only existing readings of exact hours (HH:00:00) and hour consumption calculated
readings_H = readings_2021[readings_2021.Minute == 0].copy()
readings_H['Consumo'] = readings_H.groupby('Local')['Leitura'].diff()


################################################################################
# MISSING READINGS
################################################################################

# [ADD column "missing"]
"""
Missing
== 0 row with real reading
== 1 readings to fill with data engineering (info from nearest readings will \
     be compiled)
"""
# "missing" = 0 to all real readings at exact hours (HH:00:00)
readings_H['missing'] = 0

local_lst = readings_H.Local.unique() # 483
readings_H.sort_values(by=['Local', 'Data/Hora'], inplace=True)
readings_H.set_index(['Data/Hora'], drop=False, inplace=True)

# Create reg_H with existing and missing hour readings (the latter to fill up next)
# - "missing" = 1 for hours with no reading
# - filling missing values on ["Local", "missing"] columns
reg_H = pd.DataFrame()
for l in tqdm(local_lst):
    df_aux = readings_H[readings_H.Local == l].asfreq(freq='H')
    values = {"Local": l, "missing": 1}
    df_aux.fillna(value=values, inplace=True)
    reg_H = pd.concat([reg_H, df_aux])

reg_H.Local = reg_H.Local.astype(int)
reg_H = reg_H.assign(Year=lambda x: x.index.year,
                     Month=lambda x: x.index.month,
                     Day=lambda x: x.index.day,
                     Hour=lambda x: x.index.hour,
                     Minute=lambda x: x.index.minute)
reg_H['Data/Hora'] = reg_H.index

del readings_H

# reg_H.missing.value_counts()
"""
missing count:
0.0    4225412
1.0       6151
"""
#%%
reg_H.isnull().sum()

#%%
reg_H.reset_index(drop=True, inplace=True)
reg_H.sort_values(by=['Local', 'Data/Hora'], inplace=True)

#%%
## Fetch nearest readings info ##
# [ADD columns 
#   "Hora_pos", 
#   "miss_ant", "miss_pos", 
#   "Leitura_ant", "Leitura_pos", 
#   "Leitura_diff"
# ]
# [FILL NaNs on "missing" = 1: columns 
#   "Hora_ant", 
#   "consec_H"
# ]
df_dict = reg_H.to_dict('records')
for row in tqdm(df_dict):
    if row['missing'] == 1:
        hora_ant = readings_2021[(readings_2021.Local == row['Local']) & (readings_2021['Data/Hora'] < row['Data/Hora'])].iloc[-1]
        hora_pos = readings_2021[(readings_2021.Local == row['Local']) & (readings_2021['Data/Hora'] > row['Data/Hora'])].iloc[0]
        row['Hora_ant'] = hora_pos['Hora_ant']
        row['miss_ant'] = ((row['Data/Hora'] - hora_pos['Hora_ant']) / np.timedelta64(15, 'm')) - 1
        row['Leitura_ant'] = hora_ant['Leitura']
        row['consec_H'] = row['miss_ant']*0.25
        
        row['Hora_pos'] = hora_pos['Data/Hora']
        row['miss_pos'] = ((hora_pos['Data/Hora'] - row['Data/Hora']) / np.timedelta64(15, 'm')) - 1
        row['Leitura_pos'] = hora_pos['Leitura']
reg_H = pd.DataFrame.from_dict(df_dict)
reg_H['Leitura_diff'] = reg_H['Leitura_pos'] - reg_H['Leitura_ant']

del df_dict

#%%
# FILL column "Leitura" with feature imputation
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
# Zero changes
# When there is no changes between the previous and next readings, "Leitura"
# is filled with their reading value (real value)
zero_changes = reg_H[(reg_H.missing == 1) & (reg_H.Leitura_diff == 0)].index
print(len(zero_changes))

reg_H.loc[zero_changes, 'Leitura'] = reg_H.loc[zero_changes, 'Leitura_ant']
reg_H.loc[zero_changes, 'missing'] = -2

del zero_changes
#%%
# EXPORT REG_H | STAGE 1 #
'''
4231563 entries

 #   Column           Dtype         
---  ------           -----         
 0   Data/Hora        datetime64[ns]
 1   Local            int64         
 2   Leitura          float64       
 3   miss_ant         float64       
 4   miss_pos         float64       
 5   Hora_ant         datetime64[ns]
 6   Hora_pos         datetime64[ns]
 7   Leitura_ant      float64       
 8   Leitura_pos      float64       
 9   Leitura_diff     float64       
 10  Consumo          float64       
 11  consec_H         float64       
 12  consec_D         float64       
 13  Day              int64         
 14  Month            int64         
 15  Year             int64         
 16  Hour             int64         
 17  Minute           int64         
 18  missing          float64       
 19  consec_miss_reg  float64  
'''
# column reorder
# [DEL "filename" column]
reg_H = reg_H[['Data/Hora', 'Local', 'Leitura',
               'miss_ant', 'miss_pos',
               'Hora_ant', 'Hora_pos', 
               'Leitura_ant', 'Leitura_pos', 'Leitura_diff', 
               'Consumo', 'consec_H', 'consec_D',
               'Day', 'Month', 'Year', 'Hour', 'Minute', 
               'missing', 'consec_miss_reg']].copy()

#%%
reg_H.sort_values(by=['Local', 'Data/Hora'], inplace=True)
reg_H.reset_index(drop=True, inplace=True)
reg_H.to_feather('Data/_feathers/reg_H_stage_1')

#%%
# IMPORT REG_H | STAGE 1 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_1')

#%%

################################################################################
#  DATA ANALYSIS in data_analysis.py  #  STAGE 1  #
################################################################################

# DROP Locais com +50 (+1%) "missing values"
# nº de registos (Hora) em falta
n_miss = reg_H[(reg_H.missing == 1)].groupby(by=['Local']).count()
locals_miss_H = n_miss.sort_values(by=['Data/Hora'], ascending=False)
locals_gt50_miss = locals_miss_H[locals_miss_H['Data/Hora'] > 50].index

total = 0
for local in locals_gt50_miss:
    remv = reg_H[reg_H.Local == local]
    print(str(local) + ' -> ' + str(len(remv)) + ' registos removidos')
    total += len(remv)
    reg_H.drop(remv.index, inplace=True)

print('Total = ' + str(total) + ' registos removidos')
print('\nRestam ' + str(reg_H.Local.nunique()) + ' locais')

del n_miss, locals_miss_H, locals_gt50_miss
#%%
'''
Missing
-2.0       1493
 0.0    4174846
 1.0       2658

'''
#reg_H.missing.value_counts()

#%%
# Flagging those with 1H or +1H of missing values in both sides (after and before)
# with "missing" = 2
plus1H_both = reg_H[(reg_H.missing == 1) 
                  & ((reg_H.miss_ant >= 3) & (reg_H.miss_pos >= 3))]

print('"Removidos"/descartados ' + str(len(plus1H_both.index)) + ' registos')
reg_H.loc[plus1H_both.index, 'missing'] = 2

#%%
'''
Missing
-2.0       1493
 0.0    4174846
 1.0        582
 2.0       2076

'''
#reg_H.missing.value_counts(sort=False)

#%%
# EXPORT REG_H | STAGE 2 #
reg_H.sort_values(by=['Local', 'Data/Hora'], inplace=True)
reg_H.reset_index(drop=True, inplace=True)

reg_H.to_feather('Data/_feathers/reg_H_stage_2')

#%%

################################################################################
#  DATA ANALYSIS in data_analysis.py  #  STAGE 2  #
################################################################################

# optimizável
# "missing" = -1
df_dict = reg_H.to_dict('records')

for row in tqdm(df_dict):
    if row['missing'] == 1:
        ant_dist = row['Data/Hora'] - row['Hora_ant'] 
        pos_dist = row['Hora_pos'] - row['Data/Hora']
        #mid_dist = (ant_dist + pos_dist)/2
        for minutes in [15, 30, 45]:
            # previous and next readings at same time distance (linear interpolation)
            if (ant_dist == datetime.timedelta(minutes=minutes)) & (pos_dist == datetime.timedelta(minutes=minutes)):
                row['Leitura'] = (row['Leitura_pos'] + row['Leitura_ant']) / 2
                row['missing'] = -1
            elif minutes == 45:
                break
            # 15 or 30 min from last reading (fill with that reading)
            elif ant_dist == datetime.timedelta(minutes=minutes):
                row['Leitura'] = row['Leitura_ant']
                row['missing'] = -1
            # 15 or 30 min from next reading (fill with that reading)
            elif pos_dist == datetime.timedelta(minutes=minutes):
                row['Leitura'] = row['Leitura_pos']
                row['missing'] = -1
reg_H = pd.DataFrame.from_dict(df_dict)

del df_dict

#%%

################################################################################
#  DATA ANALYSIS in data_analysis.py  #  STAGE 3  #
################################################################################

# Cálculo do Consumo
'''
O consumo de uma determinada hora corresponde à diferença de leitura entre essa 
hora e a próxima.
Ex.:
    Consumo das 19H corresponde ao consumo desde as 19H às 20H
'''
reg_H.sort_values(by=['Local', 'Data/Hora'], inplace=True)
reg_H['Consumo'] = reg_H.groupby('Local')['Leitura'].diff().shift(-1)

# remoção dos registos de '2022-01-01 00:00:00' (477, um por Local)
reg_H.drop(reg_H[reg_H['Data/Hora'] == '2022-01-01 00:00:00'].index, inplace=True)

#%%
# EXPORT REG_H | STAGE 3 #
#reg_H.reset_index(drop=True, inplace=True)
#reg_H.to_feather('Data/_feathers/reg_H_stage_3')

#%%
# IMPORT REG_H | STAGE 3 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_3')

#%%
###################################
# CONSUMOS EXCESSIVOS E NEGATIVOS
###################################

# Consumo superiores ao possível para o respectivo calibre
cadastro = pd.read_feather('Data/_feathers/df_cadastro')
cadastro = cadastro[cadastro.Local.isin(reg_H.Local.unique())]

#%%
# [ADD column "Caudal Max"]
reg_H = reg_H.join(cadastro.set_index('Local'), on='Local')
reg_H['Caudal Max'] *= 1000

del cadastro

#%%
# Remoção dos Locais com registos negativos e excessivos
# 477 - 134 = 343 Locais restantes
neg_exc_locals = reg_H[(reg_H.Consumo < 0) | (reg_H.Consumo > reg_H['Caudal Max'])].Local.unique()

sum = 0
for l in neg_exc_locals:
    idx = reg_H[reg_H.Local == l].index
    len_l = len(idx)
    sum += len_l
    print(f'Local {l} => removidos {len_l} registos')
    reg_H.drop(idx, inplace=True)
print(f'\nRemovidos {sum} registos no total de {len(neg_exc_locals)} locais.')

#%%
# EXPORT REG_H | STAGE 4 #
#reg_H.reset_index(drop=True, inplace=True)
#reg_H.to_feather('Data/_feathers/reg_H_stage_4')

#%%
# IMPORT REG_H | STAGE 4 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_4')

#%%
# Drop do Local 831069 que apenas tem consumo 0
reg_H.drop(reg_H[reg_H.Local == 831069].index, inplace=True)

#%%
# EXPORT REG_H | STAGE 5 #
#reg_H.reset_index(drop=True, inplace=True)
#reg_H.to_feather('Data/_feathers/reg_H_stage_5')

#%%
# IMPORT REG_H | STAGE 5 #
reg_H = pd.read_feather('Data/_feathers/reg_H_stage_5')

#%%
#reg_H.to_csv('Data/_feathers/reg_H_final.csv', index=False)

#%%
reg_H.info()

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
# See if there are any duplicated indexes
reg_H[reg_H.index.duplicated(keep='False')]

#%%
# Remove days without readings in the first hour of the next day
# (days wich is impossible to calculate the consumption of the 23rd hour)

from datetime import timedelta
df_aux = reg_H[(reg_H.Hour == 00) & (reg_H.Leitura.isna())].copy(deep=True)
days_to_remove0 = [(lc, ((dt - timedelta(hours=1)).month), ((dt - timedelta(hours=1)).day)) for lc, dt in zip(df_aux['Local'], df_aux['Data/Hora'])]

# Removing days with missing values (i.e. missing > 0)

days_to_remove1 = reg_H[reg_H.missing > 0].groupby(by=['Local', 'Month', 'Day']) \
                                         .count().index

reg_F = reg_H.copy()
#%%

for (local, month, day) in days_to_remove0:
    reg_F.drop(reg_F[(reg_F.Local == local) 
                    & (reg_F.Month == month) 
                    & (reg_F.Day == day)].index, inplace=True)
    print(f"Removed day {day} of month {month} from local {local}")

#%%
for (local, month, day) in days_to_remove1:
    reg_F.drop(reg_F[(reg_F.Local == local) 
                    & (reg_F.Month == month) 
                    & (reg_F.Day == day)].index, inplace=True)
    print(f"Removed day {day} of month {month} from local {local}")

#%%
reg_F.reset_index(drop=True, inplace=True)

#%%
'''
-2.0       1203
-1.0         61
 0.0    2987168
'''
reg_F.missing.value_counts()

#%%
# [ADD column 'Y_day' - day of the year]
reg_F['Y_day'] = reg_F['Data/Hora'].apply(lambda x: x.timetuple().tm_yday)

#%%
# All remaing days are full (24 hours)?
(reg_F.groupby(by=['Local', 'Y_day']).count()[['Data/Hora']] == 24).all()

#%%
reg_F.groupby('Local')['Y_day'].nunique().sort_values(ascending=True)

#%%
reg_F.groupby(by=['Local', 'Month', 'Day']).count()


#%%
#####################################################
#                                                   #
#   CONVERSION TO TIMESERIES [RAW AND NORMALIZED]   #
#                                                   #
#####################################################

locals_, dias_, dias_semana_, mes_, ts_, ts_norm_ = [], [], [], [], [], []
local_list = reg_F.Local.unique()
for l in tqdm(local_list):
    aux = reg_F.loc[reg_F.Local == l].copy()
    aux['Dia'] = aux['Data/Hora'].apply(lambda x: x.timetuple().tm_yday)
    aux['Dia_da_semana'] = aux['Data/Hora'].apply(lambda x: (x.dayofweek + 1) % 7) # para manter dom=0, seg=1, ..
    
    day_list = aux.Dia.unique()
    n_dias = len(day_list)

    for d in day_list:
        locals_.append(l)
        dias_.append(d)
        dias_semana_.append(aux[aux.Dia == d]['Dia_da_semana'].iloc[0])
        mes_.append(aux[aux.Dia == d]['Month'].iloc[0])
        ts_.append(np.resize(np.array(aux[aux.Dia == d]['Consumo']), (24,1)))
        
        if len(aux[aux.Dia == d]['Consumo']) != 24:
            print("FALHA menos de 24: %i %i" %(l,d))

        cons_dia = (aux[aux.Dia == d]['Consumo']).sum(skipna=True)
        if cons_dia:
            ts_norm_.append(np.resize(np.array(aux[aux.Dia == d]['Consumo']/cons_dia), (24,1)))
        else:
            ts_norm_.append(np.resize(np.array([0] * 24), (24,1)))
"""
result = b and a / b or 0  # a / b
"""

#%%
np.isnan(ts_).sum()

#%%
print(f"Lenghts:                                                                \
      \nLocal {len(locals_)};                                                   \
      \nDia_da_semana {len(dias_semana_)};                                      \
      \nMes {len(mes_)};                                                        \
      \nTime series {len(ts_)};                                                 \
      \nTime series norm {len(ts_norm_)}                                        \
      \nDia {len(dias_)}")


#%%
# RAW (NOT NORMALIZED)
dfnotnor = pd.DataFrame({'Local': locals_,
                   'Dia_da_semana': dias_semana_,
                   'Mes': mes_,
                   'Time series':ts_,
                   'Dia': dias_})

# NORMALIZED
dfnor = pd.DataFrame({'Local': locals_,
                      'Dia_da_semana': dias_semana_,
                      'Mes': mes_,
                      'Time series':ts_norm_,
                      'Dia': dias_})

#%%
# EXPORT NORMALIZED AND NOT NORMALIZED DATAFRAMES WITH TIME SERIES
dfnotnor.to_csv(r'data/dfnotnor.csv')
dfnotnor.to_pickle("data/dfnotnor.pkl")  # keeps 'Time series' as a array

dfnor.to_csv(r'data/dfnor.csv')
dfnor.to_pickle("data/dfnor.pkl") 


#%%

reg_F.info()
# %%
# see all rows of data frame
pd.set_option('display.max_rows', None)
reg_F.sort_values(by=['Consumo'], ascending=False) [['Local', 'Data/Hora', 'Consumo']]
# %%
# See all records from a specific day
reg_F[(reg_F.Local == 839205) & (reg_F.Month == 8) & (reg_F.Day == 5)][['Local', 'Data/Hora', 'Consumo']]

# %%
reg_F