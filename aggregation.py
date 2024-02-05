#%%
import pickle
from utils_clustering import *

'''
# GDC (Global Daily Consumption) #
Each day hour is averaged across all households for each day of the 12 months.

# HDC (Household Daily Consumption) #
One daily average time series is created for each household by averaging each 
day hour throughout the 12 months.

# HMDC (Household Month-Daily Consumption) #
Same as HDC but additional distinction for each of the 12 months of the year.

# HWDC (Household Week-Daily Consumption) #
Same as HDC but additional distinction for each of the seven days of a week.

# HWMDC (Household Week-Month-Daily Consumption) #
Same as HDC but additional distinction for each day of week of each month.

# HSDC (Household Season-Daily Consumption) #
Same as HDC but additional distinction for each of the four seasons of the year.

# DC (Daily Consumption) #
All time series.


TIME SERIES SHAPES
ts_gdc   (365, 24, 1)    # 365 days
ts_hdc   (342, 24, 1)    # 342 locals
ts_hmdc  (4104, 24, 1)   # 342 locals * 12 months
ts_hwdc  (2394, 24, 1)   # 342 locals * 7 days of week
ts_hwmdc (28728, 24, 1)  # 342 locals * 7 days of week * 12 months
ts_hsdc  (1368, 24, 1)   # 342 locals * 4 seasons
ts_dc    (124517, 24, 1) # 342 locals * 365 days  [124517 - not all days]

DATAFRAMES 
df_gdc   365 rows x 5 columns (avg1, avg2, avg3, avg4, std)
df_hdc   342 rows
df_hmdc  4104 rows
df_hwdc  2394 rows
df_hwmdc 28728 rows
df_hsdc  1368 rows
df_dc    124830 rows


# APPROACHES #
    * APPROACH 1 - Time-series clustering of hourly consumption.
    * APPROACH 2 - Clustering based on five features: the relative average of
normalised hourly consumption in a set of defined for four time periods through
the day and the standard deviation of daily water consumption

'''
# Approach 1 intervals
intervals = [0,8,11,17,24]

#%%
#                                   RAW DATA                                   #
################################################################################

df = pd.read_csv(r'Data/dfnotnor.csv')
df_pkl = pd.read_pickle("Data/dfnotnor.pkl")

#%%
# GENERATING TIME SERIES AND DATAFRAMES FOR EACH AGGREGATION APPROACH  #
########################################################################

# HDC, GDC, HWDC and HMDC Time Series and DataFrames
ts_gdc, df_gdc = gdc_pkl(df_pkl, intervals)
ts_hdc, df_hdc = hdc_pkl(df_pkl, intervals)
ts_hmdc, df_hmdc = hmdc(df, intervals)
ts_hwdc, df_hwdc = hwdc(df, intervals)

#%%
#  DC, HSDC and HWMDC Time Series and DataFrames
ts_dc, df_dc = dc(df_pkl, intervals)
ts_hsdc, df_hsdc = hsdc(df_pkl, intervals)
ts_hwmdc, df_hwmdc = hwmdc(df_pkl, intervals)


#%%
#                   OBJECTS SAVE                   #
#                                                  #
#         To avoid repeated function calls         #
####################################################

# GDC, HDC, HMDC and HWDC Time Series and DataFrames:
with open('Data/ts_df.pkl', 'wb') as f:
    pickle.dump([ts_gdc, df_gdc,
                 ts_hdc, df_hdc,
                 ts_hmdc, df_hmdc,
                 ts_hwdc, df_hwdc], f)

#%%
# DC, HSDC and HWMDC Time Series and DataFrames:
with open('Data/ts_df_2.pkl', 'wb') as f:
    pickle.dump([ts_dc, df_dc,
                 ts_hsdc, df_hsdc,
                 ts_hwmdc, df_hwmdc], f)


#%%
#                   OBJECTS LOAD                   #
####################################################

# GDC, HDC, HMDC and HWDC Time Series and DataFrames:
with open('Data/ts_df.pkl', 'rb') as f:
    ts_gdc, df_gdc, ts_hdc, df_hdc, ts_hmdc, df_hmdc, ts_hwdc, df_hwdc = pickle.load(f)

#%%
# DC, HSDC and HWMDC Time Series and DataFrames:
with open('Data/ts_df_2.pkl', 'rb') as f:
    ts_dc, df_dc, ts_hsdc, df_hsdc, ts_hwmdc, df_hwmdc = pickle.load(f)


# %%
#                               NORMALIZED DATA                                #
################################################################################

df_norm = pd.read_csv(r'data/dfnor.csv')
df_norm_pkl = pd.read_pickle("data/dfnor.pkl")

#%%
# GENERATING TIME SERIES AND DATAFRAMES FOR EACH AGGREGATION APPROACH  #
########################################################################

# HDC, GDC, HWDC and HMDC Time Series and DataFrames
ts_gdc_norm, df_gdc_norm = gdc_pkl(df_norm_pkl, intervals)
ts_hdc_norm, df_hdc_norm = hdc_pkl(df_norm_pkl, intervals)
ts_hmdc_norm, df_hmdc_norm = hmdc(df_norm, intervals)
ts_hwdc_norm, df_hwdc_norm = hwdc(df_norm, intervals)

#%%
#  DC, HSDC and HWMDC Time Series and DataFrames
ts_dc_norm, df_dc_norm = dc(df_norm_pkl, intervals)
ts_hsdc_norm, df_hsdc_norm = hsdc(df_norm_pkl, intervals)
ts_hwmdc_norm, df_hwmdc_norm = hwmdc(df_norm_pkl, intervals)


#%%

#                   OBJECTS SAVE                   #
#                                                  #
#         To avoid repeated function calls         #
####################################################

# GDC, HDC, HMDC and HWDC Time Series and DataFrames:
with open('Data/ts_df_norm.pkl', 'wb') as f:
    pickle.dump([ts_gdc_norm, df_gdc_norm, 
                 ts_hdc_norm, df_hdc_norm, 
                 ts_hmdc_norm, df_hmdc_norm, 
                 ts_hwdc_norm, df_hwdc_norm], f)

#%%
# DC, HSDC and HWMDC Time Series and DataFrames:
with open('Data/ts_df_norm2.pkl', 'wb') as f:
    pickle.dump([ts_dc_norm, df_dc_norm,
                 ts_hsdc_norm, df_hsdc_norm,
                 ts_hwmdc_norm, df_hwmdc_norm], f)


#%%
####################################################
#                   OBJECTS LOAD                   #
####################################################
#%%
# GDC, HDC, HMDC and HWDC Time Series and DataFrames:
with open('Data/ts_df_norm.pkl', 'rb') as f:
    ts_gdc_norm, df_gdc_norm, ts_hdc_norm, df_hdc_norm, ts_hmdc_norm, df_hmdc_norm, ts_hwdc_norm, df_hwdc_norm = pickle.load(f)

#%%
# DC, HSDC and HWMDC Time Series and DataFrames:
with open('Data/ts_df_2.pkl', 'rb') as f:
    ts_dc_norm, df_dc_norm, ts_hsdc_norm, df_hsdc_norm, ts_hwmdc_norm, df_hwmdc_norm = pickle.load(f)


#%%

####################################################
#                    ANALYSIS                      #
####################################################

#%%
df_norm.drop_duplicates(subset=['Local'], inplace=False).Mes.value_counts()

#%%
x=0
for l in df_norm.Local.unique():
    r = df_norm[df_norm.Local == l].Mes.is_monotonic
    if r == False:
        print("Local %s is not monotonic" %(l))
        x+=1
if x==0:
    print("All locals are monotonic")


#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_norm.Local.value_counts().sort_values(ascending=False))
