import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import datetime

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tslearn.clustering import silhouette_score as ts_sil
from tslearn.clustering import TimeSeriesKMeans
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import single, complete, average, ward, median, centroid, weighted, dendrogram
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import fcluster
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


RANDOM_SEED = 42

sns.set_style("darkgrid")


from functools import wraps
import time

# https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



'''
APPROACH 1 - Time-series clustering of hourly consumption.
APPROACH 2 - Clustering based on five features: the relative average of
normalised hourly consumption in a set of defined for four time periods through
the day and the standard deviation of daily water consumption
'''




#######################################################
#                         HDC                         #
#######################################################
'''
HDC (Household Daily Consumption)
    One daily average time series is created for each household by 
    averaging each day hour throughout the 12 months.
'''

# HDC func version #
def hdc(df, intervals):

    n_locals = df.Local.nunique()
    n_days = df.Dia.nunique()
    print(f'n_locals = {n_locals}; n_days = {n_days}')

    n_interv = len(intervals)
    
    # DataFrame inicialization
    column_names = ["Local"]
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_r = pd.DataFrame(columns=column_names)

    alllist_HDC = []
    Locals_HDC = df['Local'].unique()
    for j in range(n_locals):  # 342
        subdf = df.loc[df['Local'] == Locals_HDC[j]]
        alllist = []
        avts = []
        for k in range(n_days):  # 365
            list1 = []
            sub1 = subdf['Time series'].tolist()[k].split('\n')
            for i in range(24):
                elem1 = sub1[i].replace('[', '')
                elem2 = elem1.replace(']', '')
                numb = float(elem2)
                list1.append(numb)
            alllist.append(list1)
        sum1 = [sum(x) for x in zip(*alllist)]
        for n in range(24):
            avts.append(sum1[n]/n_days)
        ts_new = np.array(avts)
        ts_new.resize(24, 1)
        alllist_HDC.append(ts_new)

        # construction of new register (one per Local)
        local_reg = []
        local_reg.append(Locals_HDC[j])
        st = 0
        for i in range(n_interv-1):
            init = intervals[i]
            end = intervals[i+1]
            av = sum(avts[init:end])/(end-init)
            local_reg.append(av)
            st += np.std(avts[init:end])
        local_reg.append(st/(n_interv-1))
        df_r.loc[j] = local_reg

    df_HDC = df_r.drop(["Local"], axis='columns')
    ts_HDC = np.array(alllist_HDC)

    return ts_HDC, df_HDC

# HDC Pickle version #
def hdc_pkl(df, intervals):
    n_interv = len(intervals)
    
    # DataFrame inicialization
    column_names = ["Local"]
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_r = pd.DataFrame(columns=column_names)

    alllist_HDC = []
    Locals_HDC = df['Local'].unique()
    for j in range(len(Locals_HDC)):
        subdf = df.loc[df['Local'] == Locals_HDC[j]]
        local_matrix = np.stack(subdf['Time series'].values).reshape(len(subdf),24)
        
        # The arithmetic mean is the sum of the non-NaN elements along the axis
        #  divided by the number of non-NaN elements.
        avg_ts = np.nanmean(local_matrix, axis=0, dtype=np.float64)  # shape (24,1)
        avg_ts = np.array(avg_ts)
        avg_ts.resize(24,1)
       
        alllist_HDC.append(avg_ts)  # (#locals, 24, 1)

        # [DataFrame] construction of new register (one per Local)
        local_reg = []
        local_reg.append(Locals_HDC[j])
        st = 0
        for i in range(n_interv-1):
            init = intervals[i]
            end = intervals[i+1]
            #av = sum(avg_ts[init:end])/(end-init)
            av = np.mean(avg_ts[init:end])
            local_reg.append(av)
            st += np.std(avg_ts[init:end])
        local_reg.append(st/(n_interv-1))
        df_r.loc[j] = local_reg

    df_HDC = df_r.drop(["Local"], axis='columns')
    ts_HDC = np.array(alllist_HDC)

    return ts_HDC, df_HDC


#######################################################
#                         GDC                         #
#######################################################
'''
GDC (Global Daily Consumption)
    Each day hour is averaged across all households for each day of 
    the 12 months.
'''

# GDC DF version #
def gdc(df, intervals):
    n_interv = len(intervals)
    
    # DataFrame inicialization
    column_names = ["Dia"]
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_r = pd.DataFrame(columns=column_names)

    alllist_GDC = []
    days = df['Dia'].unique()
    n_days = len(days)
    n_locals = df['Local'].nunique()

    for j in range(n_days):  # 365
        subdf = df.loc[df['Dia'] == days[j]]
        alllist1 = []
        avg_ts = []
        for k in range(n_locals):  
            list1 = []
            sub1 = subdf['Time series'].tolist()[k].split('\n')
            for i in range(24):
                elem1 = sub1[i].replace('[', '')
                elem2 = elem1.replace(']', '')
                numb = float(elem2)
                list1.append(numb)
            alllist1.append(list1)
        
        sum1 = [sum(x) for x in zip(*alllist1)]
        for n in range(24):
            avg_ts.append(sum1[n]/n_locals)
        ts_new = np.array(avg_ts)
        ts_new.resize(24, 1)
        alllist_GDC.append(ts_new)

        # construction of new register (one per Day)
        reg = []
        reg.append(days[j])
        st = 0
        for i in range(n_interv-1):
            init = intervals[i]
            end = intervals[i+1]
            av = sum(avg_ts[init:end])/(end-init)
            reg.append(av)
            st += np.std(avg_ts[init:end])
        reg.append(st/(n_interv-1))
        df_r.loc[j] = reg

    #print(df_r)
    df_GDC = df_r.drop(["Dia"], axis='columns')
    ts_GDC = np.array(alllist_GDC)

    return ts_GDC, df_GDC

# GDC pickle version #
def gdc_pkl(df, intervals):
    n_interv = len(intervals)
    
    # DataFrame inicialization
    column_names = ["Dia"]
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_r = pd.DataFrame(columns=column_names)

    alllist_GDC = []
    days = df['Dia'].unique()
    n_dias = len(days)
    for j in range(n_dias):  # 365 dias
        avg_ts = []
        subdf = df.loc[df['Dia'] == days[j]]
        matrix = np.stack(subdf['Time series'].values).reshape(len(subdf),24)
        avg_ts = np.nanmean(matrix, axis=0, dtype=np.float64)

        ts_new = np.array(avg_ts)
        ts_new.resize(24, 1)
        alllist_GDC.append(ts_new)

        # [DataFrame] construction of new register (one per Local)
        reg = []
        reg.append(days[j])
        st = 0
        for i in range(n_interv-1):
            init = intervals[i]
            end = intervals[i+1]
            #av = sum(avg_ts[init:end])/(end-init)
            av = np.mean(avg_ts[init:end])
            reg.append(av)
            st += np.std(avg_ts[init:end])
        reg.append(st/(n_interv-1))
        df_r.loc[j] = reg

    df_GDC = df_r.drop(["Dia"], axis='columns')
    ts_GDC = np.array(alllist_GDC)

    return ts_GDC, df_GDC


#######################################################
#                         HWDC                        #
#######################################################
'''
HWDC (Household Week-Daily Consumption)
    Same as HDC but additional distinction for each of the seven days
    of a week.
'''

# HWDC DF version #
def hwdc(df, intervals):
    n_interv = len(intervals)
    
    # DataFrame inicialization
    column_names = ["Local", "Dia_da_semana"]
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_r = pd.DataFrame(columns=column_names)

    alllist_HWDC = []

    # Pandas docs: Uniques are returned in order of appearance, it does NOT sort
    # but df 'Local' column is sorted [df.Local.is_monotonic = True]
    Locals_HWDC = df['Local'].unique()
    n_locals = len(Locals_HWDC)
    Days = []
    nrow = 0
    for j in range(n_locals):
        subdf1 = df.loc[df['Local']==Locals_HWDC[j]]

        # For each Local, in order of appearance, we have the 7 days of the week
        # but the weekdays are appended in order (0,1,2,3,4,5,6), so df must be
        # ordered in the same way if we want to have the same order, for time series
        # identification purposes.
        for wd in range(7):
            subdf2 = subdf1.loc[subdf1['Dia_da_semana']==wd]
            Days.append(wd)
            alllist1 = []
            avg_ts = []
            for k in range(len(subdf2)):
                list1 = []
                sub1 = subdf2['Time series'].tolist()[k].split('\n')
                for i in range(24):
                    elem1=sub1[i].replace('[','')
                    elem2=elem1.replace(']','')
                    numb=float(elem2)
                    list1.append(numb)
                alllist1.append(list1)

            matrix = np.stack(alllist1).reshape(len(subdf2),24)
            avg_ts = np.nanmean(matrix, axis=0, dtype=np.float64)
            ts_new = np.array(avg_ts)
            ts_new.resize(24, 1)

            alllist_HWDC.append(ts_new)

            # construction of new register
            reg = []
            reg.append(Locals_HWDC[j])
            reg.append(wd)
            st = 0
            for i in range(n_interv-1):
                init = intervals[i]
                end = intervals[i+1]
                av = np.mean(avg_ts[init:end])
                reg.append(av)
                st += np.std(avg_ts[init:end])
            reg.append(st/(n_interv-1))
            df_r.loc[nrow] = reg
            nrow += 1

    df_HWDC = df_r.drop(["Local", "Dia_da_semana"], axis='columns')
    ts_HWDC = np.array(alllist_HWDC)

    return ts_HWDC, df_HWDC


#######################################################
#                         HMDC                        #
#######################################################
'''
HMDC (Household Month-Daily Consumption)
    Same as HDC but additional distinction for each of the seven days of a week.
'''

# HMDC DF version #
def hmdc(df, intervals):
    n_interv = len(intervals)
    
    # DataFrame inicialization
    column_names = ["Local", "Mes"]
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_r = pd.DataFrame(columns=column_names)

    alllist_HMDC = []
    Locals_HMDC = df['Local'].unique()
    n_locals = len(Locals_HMDC)
    Months_HDMC = df['Mes'].unique()
    Months = []
    nrow = 0
    # For each Local, in order of appearance
    # df Locals are already ordered
    for j in range(n_locals):
        subdf1=df.loc[df['Local']==Locals_HMDC[j]]
        # For each Local, in order of appearance, we have the 12 months of 
        # the year. Also ordered (1-12).
        for m in Months_HDMC:
            subdf2 = subdf1.loc[subdf1['Mes']==m]
            Months.append(m)
            alllist1=[]
            avg_ts=[]
            for k in range(len(subdf2)):
                list1 = []
                sub1 = subdf2['Time series'].tolist()[k].split('\n')
                for i in range(24):
                    elem1=sub1[i].replace('[','')
                    elem2=elem1.replace(']','')
                    numb=float(elem2)
                    list1.append(numb)
                alllist1.append(list1)
            
            matrix = np.stack(alllist1).reshape(-1,24)
        
            avg_ts = np.nanmean(matrix, axis=0, dtype=np.float64)
            ts_new = np.array(avg_ts)
            ts_new.resize(24, 1)

            alllist_HMDC.append(ts_new)

            # construction of new register
            reg = []
            reg.append(Locals_HMDC[j])
            reg.append(m)
            st = 0
            for i in range(n_interv-1):
                init = intervals[i]
                end = intervals[i+1]
                av = sum(avg_ts[init:end])/(end-init)
                reg.append(av)
                st += np.std(avg_ts[init:end])
            reg.append(st/(n_interv-1))
            df_r.loc[nrow] = reg
            nrow += 1

    df_HMDC = df_r.drop(["Local", "Mes"], axis='columns')
    ts_HMDC = np.array(alllist_HMDC)

    return ts_HMDC, df_HMDC


#######################################################
#                         HWMDC                        #
#######################################################
'''
HWMDC (Household Week-Month-Daily Consumption)
Same as HDC but additional distinction for each day of week of each month.

342 Locals * 7 Days of week * 12 Months = 28 728 records

'''

def hwmdc(df, intervals):
    '''
    Create two dataframes, one in approach 1 and another in approach 2.
    Each record of the dataframe corresponds to the average of each day of the 
    week, for each month, for each Local.

    Parameters
    ----------
    df : pandas dataframe
        Data frame with the raw or normalized data with the following columns:
        Local | Dia_da_semana | Mes | Time series | Dia.
        This dataframe is ordered by "Local" and then by "Dia".
    intervals : list
        List with the intervals to use on the approach 2.
    
    Returns
    -------
    df_1 : numpy array
        Array constructed in approach 1.
    df_2 : pandas dataframe
        Dataframe constructed in approach 2.
    
    '''
    # Approach 2 DataFrame inicialization
    column_names = ["Local", "Mes", "Dia_da_semana"]
    n_interv = len(intervals)
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_2 = pd.DataFrame(columns=column_names)

    a1_list = []
    row = 0
    for local in df['Local'].unique():
        df_local = df.loc[df['Local'] == local]
        for month in range(1,13):
            df_month = df_local.loc[df_local['Mes'] == month]
            for day_of_week in range(0,7):
                df_day_of_week = df_month.loc[df_month['Dia_da_semana'] == day_of_week]
                
                local_matrix = np.stack(df_day_of_week['Time series'].values).reshape(len(df_day_of_week),24)

                # The arithmetic mean is the sum of the non-NaN elements along the axis
                #  divided by the number of non-NaN elements.
                avg_ts = np.nanmean(local_matrix, axis=0, dtype=np.float64)  # shape (24,1)
                avg_ts = np.array(avg_ts)
                avg_ts.resize(24,1)

                a1_list.append(avg_ts)

                # Approach 2
                record = []
                record.append(local)
                record.append(month)
                record.append(day_of_week)

                std = 0
                for i in range(n_interv-1):
                    start = intervals[i]
                    end = intervals[i+1]
                    record.append(np.mean(avg_ts[start:end]))
                    std += np.std(avg_ts[start:end])
                record.append(std/(n_interv-1))
                df_2.loc[row] = record
                row += 1

    df_1 = np.array(a1_list)
    #df_confirm = df_2.copy(deep=True)
    df_2 = df_2.drop(columns=['Local', 'Mes', 'Dia_da_semana'])

    return df_1, df_2


#######################################################
#                         HSDC                        #
#######################################################
'''
HSDC (Household Season-Daily Consumption)
Same as HDC but additional distinction for each of the four seasons of the year.

342 Locals * 4 Seasons = 1368 records

#######################################################
2021 Seasons - Northern Hemisphere
https://www.timeanddate.com/calendar/seasons.html

Spring 20/03 - 20/06
Summer 21/06 - 21/09
Autumn 22/09 - 20/12
Winter 21/12 - 19/03

DATE => Day of the year (Dia)
* https://nsidc.org/data/user-resources/help-center/day-year-doy-calendar

20/03 => 79
20/06 => 171
21/06 => 172
21/09 => 264
22/09 => 265
20/12 => 354
21/12 => 355
19/03 => 78

#######################################################
'''

def hsdc(df, intervals):
    '''
    Create two dataframes, one in approach 1 and another in approach 2.
    Each record of the dataframe corresponds to the average of each season, 
    for each Local.

    Parameters
    ----------
    df : pandas dataframe
        Data frame with the raw or normalized data with the following columns:
        Local | Dia_da_semana | Mes | Time series | Dia.
        This dataframe is ordered by "Local" and then by "Dia".
    intervals : list
        List with the intervals to use on the approach 2.
    
    Returns
    -------
    df_1 : numpy array
        Array constructed in approach 1.
    df_2 : pandas dataframe
        Dataframe constructed in approach 2.
    
    '''
    # Approach 2 DataFrame inicialization
    column_names = ["Local", "Season"]
    n_interv = len(intervals)
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_2 = pd.DataFrame(columns=column_names)

    a1_list = []
    row = 0
    for local in df['Local'].unique():
        df_local = df.loc[df['Local'] == local]
        for season in ['spring', 'summer', 'autumn', 'winter']:
            if season == 'spring':
                df_season = df_local.loc[(df_local['Dia'] >= 79) & (df_local['Dia'] <= 171)]
            elif season == 'summer':
                df_season = df_local.loc[(df_local['Dia'] >= 172) & (df_local['Dia'] <= 264)]
            elif season == 'autumn':
                df_season = df_local.loc[(df_local['Dia'] >= 265) & (df_local['Dia'] <= 354)]
            elif season == 'winter':
                df_season = df_local.loc[((df_local['Dia'] >= 355) & (df_local['Dia'] <= 365)) | ((df_local['Dia'] >= 0) & (df_local['Dia'] <= 78))]
                
            local_matrix = np.stack(df_season['Time series'].values).reshape(len(df_season),24)

            # The arithmetic mean is the sum of the non-NaN elements along the axis
            #  divided by the number of non-NaN elements.
            avg_ts = np.nanmean(local_matrix, axis=0, dtype=np.float64)  # shape (24,1)
            avg_ts = np.array(avg_ts)
            avg_ts.resize(24,1)

            a1_list.append(avg_ts)

            # Approach 2
            record = []
            record.append(local)
            record.append(season)

            std = 0
            for i in range(n_interv-1):
                start = intervals[i]
                end = intervals[i+1]
                record.append(np.mean(avg_ts[start:end]))
                std += np.std(avg_ts[start:end])
            record.append(std/(n_interv-1))
            df_2.loc[row] = record
            row += 1

    df_1 = np.array(a1_list)
    #df_confirm = df_2.copy(deep=True)
    df_2 = df_2.drop(columns=['Local', 'Season'])

    return df_1, df_2


#######################################################
#                          DC                         #
#######################################################
'''
DC (Daily Consumption)
All time series.

342 Locals * 365 Days = 124 830 records

'''

def dc(df, intervals):
    '''
    Create two dataframes, one in approach 1 and another in approach 2.
    Each record of the dataframe corresponds to the consumption of each day, 
    for each Local.

    Parameters
    ----------
    df : pandas dataframe
        Data frame with the raw or normalized data with the following columns:
        Local | Dia_da_semana | Mes | Time series | Dia.
        This dataframe is ordered by "Local" and then by "Dia".
    intervals : list
        List with the intervals to use on the approach 2.
    
    Returns
    -------
    df_1 : numpy array
        Array constructed in approach 1.
    df_2 : pandas dataframe
        Dataframe constructed in approach 2.
    
    '''
    # Approach 2 DataFrame inicialization
    column_names = ["Local", "Dia"]
    n_interv = len(intervals)
    for i in range(n_interv-1):
        column_names.append("average"+str(i+1))
    column_names.append("std")
    df_2 = pd.DataFrame(columns=column_names)

    a1_list = []
    row = 0
    for local in df['Local'].unique():
        df_local = df.loc[df['Local'] == local]

        for day in df_local['Dia'].unique():
            df_day = df_local.loc[df_local['Dia'] == day]
            ts = np.array(df_day['Time series'].values)[0]
            a1_list.append(ts)

            # Approach 2
            record = []
            record.append(local)
            record.append(day)

            std = 0
            for i in range(n_interv-1):
                start = intervals[i]
                end = intervals[i+1]
                record.append(np.mean(ts[start:end]))
                std += np.std(ts[start:end])
            record.append(std/(n_interv-1))
            df_2.loc[row] = record
            row += 1

    df_1 = np.array(a1_list)
    #df_confirm = df_2.copy(deep=True)
    df_2 = df_2.drop(columns=['Local', 'Dia'])

    return df_1, df_2


#######################################################
#                          CV                         #
#             Coefficient of Variation                #
#######################################################
def cv(df):
    # DataFrame inicialization
    column_names = ["Local"]
    column_names.append("daily_sd") # average of daily standard deviation
    column_names.append("sd") # standard deviation of the year consumption
    column_names.append("avg") # average of daily consumption
    column_names.append("cv") # coefficient of variation
    df_r = pd.DataFrame(columns=column_names)

    for l in df.Local.unique():
        subdf = df.loc[df['Local'] == l]
        day_cons = []
        day_sd = []
        # matrix with all time series for each day
        stack_ = np.stack(subdf['Time series'].values).reshape(-1, 24)
        
        # daily consumption and standard deviation
        day_cons = np.array([sum(stack_[d,:]) for d in range(stack_.shape[0])])
        day_sd = np.array([np.std(stack_[d,:]) for d in range(stack_.shape[0])])
        
        # Construction of new register (one per Local)
        local_reg = []
        local_reg.append(int(l))

        # "daily_sd" - average of daily standard deviations
        local_reg.append(np.mean(day_sd))

        # "sd" - standard deviation of daily consumption for the year
        local_reg.append(np.std(day_cons))

        # "avg" - average of daily consumption for the year
        local_reg.append(np.mean(day_cons))

        # "cv" - coefficient of variation
        local_reg.append(np.std(day_cons)/np.mean(day_cons))


        df_r.loc[len(df_r)] = local_reg

    return df_r
              
"""
#######################################################
#                  ELBOW METHOD                       #
#######################################################

# conda install -c districtdatalabs yellowbrick
# conda install h5py
# pip install yellowbrick==1.3.post1  para resolver o erro: AttributeError: module 'h5py' has no attribute 'version'
# https://github.com/DistrictDataLabs/yellowbrick/issues/1137
# que consequentemente fez o downgrade do numpy para 1.19.5
@timeit
def elbow(
    data, 
    k_range, 
    clust_func,
    metric = None,
    metric_yb='distortion', 
    timings=False, 
    locate_elbow=True, 
    save='',
    clear_figure=True,
    **kwargs
    ):
    
    # Instantiate the clustering model and visualizer
    if clust_func == TimeSeriesKMeans:
        model = clust_func(random_state=RANDOM_SEED, metric=metric, **kwargs)
    elif clust_func == KMeans:
        model = clust_func(random_state=RANDOM_SEED, **kwargs)

    visualizer = KElbowVisualizer(model, k=k_range, metric=metric_yb, timings=timings, locate_elbow=locate_elbow)

    visualizer.fit(data)        # Fit the data to the visualizer
    
    if save != '':
        visualizer.show(outpath=save+'.png')
    # Finalize and render the figure
    else:
        # clear figure so that it doesn't show up in the next plot
        if clear_figure:
            visualizer.show(clear_figure=True)
        else:
            visualizer.show()        


def elbow_method(dfr, n_clusters, save=''):
    wcss = []

    for k in n_clusters:    
        # Run the kmeans algorithm
        km = KMeans(n_clusters=k,random_state=RANDOM_SEED)
        cluster_labels = km.fit_predict(dfr)
        centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        wcss.append(km.inertia_)

    print(f'\nWithin-Clusters Sum-of-Squares (WCSS):\n{wcss}')

    #View: Elbow Method
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Número de clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Método do cotovelo')
    plt.grid(True)

    y2 = wcss[len(wcss)-1]
    y1 = wcss[0]

    plt.plot([max(n_clusters), min(n_clusters)], [y2,y1], color='darkorange') # reta min-max
    plt.plot(n_clusters, wcss, color='tab:blue') # linha principal
    plt.plot(n_clusters, wcss, '.', color='r') # pontos

    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.xticks(n_clusters)
    #plt.show()

    if save != '':
        plt.savefig(save+'.png', dpi=250)

    plt.clf
    plt.close('all')

    x1, y1 = n_clusters[0], wcss[0]
    x2, y2 = n_clusters[-1], wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    best_k = distances.index(max(distances)) + 2
    print('By Elbow Method the best clustering is with k =', best_k)


def elbow_method_TS(ts, n_clusters, save=''):
    wcss = []

    for k in n_clusters:    
        # Run the kmeans algorithm
        km = TimeSeriesKMeans(
                        n_clusters=k, 
                        metric="euclidean", 
                        max_iter=10,
                        random_state=RANDOM_SEED)
        km.fit(ts)
        cluster_labels = km.predict(ts)
        #cluster_labels = km.fit_predict(ts)
        centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        wcss.append(km.inertia_)

    print(f'\nWithin-Clusters Sum-of-Squares (WCSS):\n{wcss}')

    #View: Elbow Method
    fig = plt.figure()
    ax5 = fig.add_subplot()
    ax5.set_xlabel('Número de clusters')
    ax5.set_ylabel('WCSS')
    ax5.set_title('Método do cotovelo')
    plt.grid(True)

    y2 = wcss[len(wcss)-1]
    y1 = wcss[0]

    plt.plot([max(n_clusters), min(n_clusters)], [y2,y1], color='darkorange') # reta min-max
    plt.plot(n_clusters, wcss, color='tab:blue') # linha principal
    plt.plot(n_clusters, wcss, '.', color='r') # pontos

    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.xticks(n_clusters)
    # plt.show()

    if save != '':
        plt.savefig(save+'.png', dpi=250)
    plt.clf
    plt.close('all')

    x1, y1 = n_clusters[0], wcss[0]
    x2, y2 = n_clusters[-1], wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    best_k = distances.index(max(distances)) + 2
    print('By Elbow Method the best clustering is with k =', best_k)


#######################################################
#                   SILHOUETTE                        #
#######################################################
@timeit
def silhouette(dfr, k_range, clust_func, metric_params=None, sil_metric=None, save_txt='', save_img='', imgs=False, **kwargs):
    '''
    Computes the silhouette score for a range of k (number of cluster) values.
    
    Parameters
    ----------
    dfr : data
        Data to be clustered.
    k_range : list
        List of k values to be tested.
    clust_func : function
        Clustering function to be used.
    metric_params : dict
        Parameters to be passed to the clustering function and the silhouette 
        metric function (if TimeSeriesKMeans is used).
    sil_metric : function
        Silhouette metric function to be used. If None, the default metric
        function is used.
    save : str
        Path to save the silhouette plot. Only used if imgs is True.
    imgs : bool
        If True, the silhouette plot is shown.
    **kwargs : dict
        Additional parameters to be passed to the clustering function.

    '''
    results = []
    for k in k_range:
        if imgs:
            fig, (ax1) = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)
            ax1.set_xlim([-0.1, 1])
            # The (k+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(dfr) + (k + 1) * 10])
        
        # Run the clustering algorithm
        km = clust_func(n_clusters=k, random_state=RANDOM_SEED, metric_params=metric_params, **kwargs)
        km.fit(dfr)
        cluster_labels = km.predict(dfr)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        if clust_func == TimeSeriesKMeans:
            # DTW is the default metric for tslearn silhouette
            silhouette_avg = ts_sil(dfr, cluster_labels, metric=sil_metric, metric_params=metric_params)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(dfr.reshape(len(dfr),24), cluster_labels)
        
        elif clust_func == KMeans:
            # Euclidean is the default metric for sklearn silhouette
            silhouette_avg = silhouette_score(dfr, cluster_labels, metric=sil_metric)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(dfr, cluster_labels)
        
        print_score = f'k = {k} => Average Silhouette Score = {round(silhouette_avg, 4)}'
        if save_txt != '':
            with open(save_txt + '.txt',"a+") as f:
                f.write(print_score + '\n')
        print(print_score)

        results.append(silhouette_avg)

        if imgs:
            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    alpha=0.7,
                )
                
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title(f"The silhouette plot for the various clusters (with  k = {k})")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette = "+str(round(silhouette_avg,2)))
            ax1.legend(loc='upper right')
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            if save_img != '':
                plt.savefig(save_img+'_'+str(k)+'.png', dpi=250)
            else:
                plt.show()

            # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
            plt.clf
            plt.close('all')

    # Print higher silhouette score
    best_score = max(results)
    best_k_idx = results.index(best_score)

    print_best_score = f'\nBest score: {best_score} with k = {k_range[best_k_idx]}\n\n'
    if save_txt != '':
        with open(save_txt + '.txt',"a+") as f:
                f.write(print_best_score)
    print(print_best_score)



# WCSS ( Within-Cluster Sum of Square ) is the sum of squared distance between 
# each point and the centroid in a cluster.
'''
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2
'''


def silhouette_method(dfr, k_range, save=''):
    for k in k_range:
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        # The (k+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dfr) + (k + 1) * 10])
        

        # Run the kmeans algorithm
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED)
        cluster_labels = km.fit_predict(dfr)
        centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dfr, cluster_labels)
        print("#clusters =", k, " => Average Silhouette Score =", round(silhouette_avg, 4))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dfr, cluster_labels)

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"The silhouette plot for the various clusters (with  k = {k})")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette = "+str(round(silhouette_avg,2)))
        ax1.legend(loc='upper right')
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        if save != '':
            plt.savefig(save+'_'+str(k)+'.png', dpi=250)

        # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.clf
        plt.close('all')

        # closes a window, which will be the current window, if not specified otherwise.
        #plt.close('all')


def silhouette_method_TS(ts, k_range, save=''):
    for k in k_range:
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        # The (k+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(ts) + (k + 1) * 10])
    

        # Run the kmeans algorithm
        km = TimeSeriesKMeans(
                        n_clusters=k, 
                        metric="euclidean", 
                        max_iter=10,
                        random_state=RANDOM_SEED)
        km.fit(ts)
        cluster_labels = km.predict(ts)
        #cluster_labels = km.fit_predict(ts)
        #centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        #TimeSeriesKMeans silhouette_score as ts_sil
        silhouette_avg = ts_sil(ts, cluster_labels, metric="euclidean")
        print("#clusters =", k, " => Average Silhouette Score =", round(silhouette_avg, 4))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(ts.reshape(len(ts),24), cluster_labels)
        #print(sample_silhouette_values)
        #print(len(sample_silhouette_values))
        
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i,
            # and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"The silhouette plot for the various clusters (with  k = {k})")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette = "+str(round(silhouette_avg,2)))
        ax1.legend(loc='upper right')
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        if save != '':
            plt.savefig(save+'_'+str(k)+'.png', dpi=250)

        plt.clf
        plt.close('all')


def silhouette_method_TS_dtw(ts, k_range, save=''):
    for k in k_range:
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        # The (k+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(ts) + (k + 1) * 10])
    

        # Run the kmeans algorithm
        km = TimeSeriesKMeans(
                        n_clusters=k, 
                        metric="dtw",
                        metric_params={'sakoe_chiba_radius': 1},
                        max_iter=10,
                        random_state=RANDOM_SEED)
        km.fit(ts)
        cluster_labels = km.predict(ts)
        #cluster_labels = km.fit_predict(ts)
        #centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        #TimeSeriesKMeans silhouette_score as ts_sil
        silhouette_avg = ts_sil(ts, cluster_labels, metric="euclidean")
        print("#clusters =", k, " => Average Silhouette Score =", round(silhouette_avg, 4))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(ts.reshape(len(ts),24), cluster_labels)
        #print(sample_silhouette_values)
        #print(len(sample_silhouette_values))
        
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i,
            # and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"The silhouette plot for the various clusters (with  k = {k})")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette = "+str(round(silhouette_avg,2)))
        ax1.legend(loc='upper right')
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        if save != '':
            plt.savefig(save+'_'+str(k)+'.png', dpi=250)

        plt.clf
        plt.close('all')
"""                

#######################################################
#                     CENTROIDS                       #
#                       MEDIA                         #
#######################################################

# GDC
# Clustering on Approach 2 but giving a 24 hours view of all days consumption 
# and its cluster centroids
def A2_clustering_A1_view(a2_df, a1_df, n_clusters, save=''):
    # Clustering using Approach 2
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    y_pred = km.fit_predict(a2_df)

    unique, counts = np.unique(y_pred, return_counts=True)
    print("Distribution of cluster elements => ", str(dict(zip(unique, counts))))

    # Centroids
    # Cluster centroids are calculated by taking the mean of daily
    # profiles that belongs to the assigned cluster number
    average_centroid=[]
    for k in range(n_clusters):
        a1_clust_match = a1_df[y_pred == k]
        average_centroid.append(sum(a1_clust_match)/len(a1_clust_match))
    print('AVERAGE_CENTROIDS:')
    print(average_centroid)

    for yi in range(n_clusters):
        plt.figure()
        #plt.subplot(n_clusters, 1, yi + 1)
        for xx in a1_df[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3, linewidth=0.5)
        plt.plot(average_centroid[yi], "r-")
        plt.xlim(0, a1_df.shape[1])
        #plt.ylim(-0.1, 0.7)
        plt.ylim(-0.01,a1_df.max()*1.3)
        plt.title("Cluster " + str(yi))

        if save != '':
            plt.savefig(save+'_'+str(yi)+'.png', dpi=250)


def A2_clust_mean_values_cluster_centroids_A1(a2_df, a1_df, n_clusters, save=''):
    # Clustering using Approach 2
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    y_pred = km.fit_predict(a2_df)

    centroids = km.cluster_centers_

    # centroids.shape = (n_clust,5)
    print(f'Centroids Approach 2: [shape {centroids.shape}]\n{centroids}')

    unique, counts = np.unique(y_pred, return_counts=True)
    clst_distrib = dict(zip(unique, counts))

    with open(save + '_logs.txt', 'w') as f:
        f.write(f"Distribution of cluster elements => {str(clst_distrib)} \n")

        print("Distribution of cluster elements => ", str(clst_distrib))

        for k in range(n_clusters):
            print(f'\nCluster {str(k)} mean values:')
            print(a2_df[y_pred == k].mean())

            f.write(f'\nCluster {str(k)} mean values:')
            f.write(a2_df[y_pred == k].mean().to_string())
            f.write('\n')
    

    # Centroids
    # Cluster centroids are calculated by taking the mean of daily
    # profiles that belongs to the assigned cluster number
    average_centroid=[]
    for k in range(n_clusters):
        a1_clust_match = a1_df[y_pred == k]
        average_centroid.append(sum(a1_clust_match)/len(a1_clust_match))

    # Cluster centroids (avg centroids of all clusters in one plot)
    plt.figure()
    for c in range(n_clusters):
        plt.plot(average_centroid[c].ravel(), alpha=0.8, label='Cluster '+str(c))
    plt.xlim(0, 24)
    plt.ylim(0, max(np.ravel(average_centroid))*1.2)
    plt.xlabel('Hour')
    plt.title("Centroids")
    plt.legend()

    if save != '':
            plt.savefig(save+'.png', dpi=250)


def TimeSeriesKMeans_centroids(ts, n_clusters, save=''):
    km = TimeSeriesKMeans(
                        n_clusters=n_clusters, 
                        metric="euclidean", 
                        max_iter=10,
                        random_state=RANDOM_SEED)
    y_pred = km.fit_predict(ts)
    centroids = km.cluster_centers_

    print(f'Centroids shape {centroids.shape}')
    print("MIN  MAX")
    print(centroids.min(), centroids.max())

    plt.figure()
    for c in range(n_clusters):
        plt.plot(centroids[c].ravel(), alpha=0.8, label='Cluster '+str(c))
    plt.xlim(0, 24)
    plt.ylim(0,centroids.max()*1.2)
    plt.xlabel('Hour')
    plt.title("Centroids")
    plt.legend()

    if save != '':
        plt.savefig(save, dpi=250)


def KMeans_centroids(ts, n_clusters, save=''):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    y_pred = km.fit_predict(ts)
    centroids = km.cluster_centers_

    print(f'Centroids shape {centroids.shape}')
    print("MIN  MAX")
    print(centroids.min(), centroids.max())

    plt.figure()
    for c in range(n_clusters):
        plt.plot(centroids[c].ravel(), alpha=0.8, label='Cluster '+str(c))
    plt.xlim(0, 24)
    plt.ylim(0,centroids.max()*1.2)
    plt.xlabel('Hour')
    plt.title("Centroids")
    plt.legend()

    if save != '':
        plt.savefig(save, dpi=250)


def centroids_plot_v2(ap1_ts, k, save, clust_func, **kwargs):
    km = clust_func(**kwargs)
    y_pred = km.fit_predict(ap1_ts)
    centroids = km.cluster_centers_

    print(f'Centroids shape {centroids.shape}')
    print("MIN  MAX")
    print(centroids.min(), centroids.max())

    plt.figure()
    for c in range(k):
        plt.plot(centroids[c].ravel(), alpha=0.8, label='Cluster '+str(c))
    plt.xlim(0, 24)
    plt.ylim(0,centroids.max()*1.2)
    plt.xlabel('Hour')
    plt.title("Centroids")
    plt.legend()

    if save != '':
        plt.savefig(save, dpi=250)


def week_month_hist(df, df_gdc, n_clusters, mode='count', save=''):
    # presents all the 365 days of the year in order with "Dia", "Dia_da_semana", "Mes" columns
    df_aux = df.drop_duplicates(subset=['Dia'])[['Dia','Dia_da_semana','Mes']]

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    labels = km.fit_predict(df_gdc)
    df_aux['Cluster'] = pd.Series(labels)

    # Days of week => Sun = 0, Sat = 6
    week_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    for c in range(n_clusters):
        clst_df = df_aux[df_aux['Cluster'] == c]
        print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
        if mode == 'perc':
            day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index()
            stat = 'percent'
            ylabel = "Percentage (%)"
        else:
            day_counts = ((clst_df['Dia_da_semana'].value_counts())).sort_index()
            stat = 'count'
            ylabel = "No. of observations"
        #day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index() 
        print((day_counts))
        
        plt.figure(figsize=(10,5))
        ax = sns.histplot(data=clst_df, x="Dia_da_semana", stat=stat, discrete=True, shrink=.8)
        plt.title("Cluster " + str(c))
        plt.xlabel("Day of Week")
        plt.xticks(range(7), week_days, rotation='horizontal')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False)
        plt.ylabel(ylabel)
        plt.ylim(0, 100)
        #plt.show()
        plt.savefig(save+'day_of_week_cluster'+str(c)+'.png', dpi=250)

    # Months
    month_dic = {
        1:  'Jan',
        2:  'Feb',
        3:  'Mar',
        4:  'Apr',
        5:  'May',
        6:  'Jun',
        7:  'Jul',
        8:  'Ago',
        9:  'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec',
    }
    df_aux['Month'] = df_aux['Mes'].replace(month_dic)

    df_aux2 = df_aux.copy()
    df_aux2['Mes'].replace(month_dic,inplace=True)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
    for c in range(n_clusters):
        clst_df = df_aux2[df_aux2['Cluster'] == c]
        print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
        if mode == 'perc':
            day_counts = ((clst_df['Mes'].value_counts(normalize=True))*100).sort_index()
            stat = 'percent'
            ylabel = "Percentage (%)"
        else:
            day_counts = ((clst_df['Mes'].value_counts())).sort_index()
            stat = 'count'
            ylabel = "No. of observations"
        
        #day_counts = ((clst_df['Mes'].value_counts(normalize=True))*100).sort_index() 
        print((day_counts))
        
        plt.figure(figsize=(10,5))
        ax = sns.histplot(data=clst_df, x="Month", stat=stat, discrete=True, shrink=.8)
        plt.title("Cluster " + str(c), fontsize=16)
        plt.xlabel("Month", fontsize=14)
        plt.xticks(range(12), months, rotation='horizontal')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False)
        plt.ylabel(ylabel, fontsize=14)
        plt.ylim(0, 100)
        #plt.show()
        plt.savefig(save+'month_cluster'+str(c)+'.png', dpi=250)


## DTW JOCLAD - HWDC - day_of_week ##
def week_month_hist_TS(df, ts, n_clusters, approach='HWDC', mode='count', save=''):
    # Adjustments on df for the HWDC approach
    if approach == 'HWDC':
        df_aux = df.sort_values(by=['Local', 'Dia_da_semana'])
        df_aux.drop_duplicates(subset=['Local', 'Dia_da_semana'], inplace=True)
        df_aux = df_aux[['Dia','Dia_da_semana']]

    if approach == 'HMDC':
        df_aux = df.sort_values(by=['Local', 'Mes'])
        df_aux.drop_duplicates(subset=['Local', 'Mes'], inplace=True)
        df_aux = df_aux[['Dia','Mes']]

    if approach == 'GDC':
        df_aux = df.drop_duplicates(subset=['Dia'])[['Dia','Dia_da_semana','Mes']]

    # Run the clustering algorithm
    km = TimeSeriesKMeans(
                    n_clusters=n_clusters, 
                    metric="dtw",
                    metric_params={'sakoe_chiba_radius': 2},
                    max_iter=10,
                    random_state=RANDOM_SEED)

    km.fit(ts)
    labels = km.predict(ts)

    # Add cluster labels to the DataFrame
    df_aux['Cluster'] = labels.tolist()

    # Show clusters distribution by day of week
    if approach not in ['HMDC']:
        # Days of week => Sun = 0, Sat = 6
        week_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        for c in range(n_clusters):
            clst_df = df_aux[df_aux['Cluster'] == c]
            print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
            if mode == 'perc':
                day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index()
                stat = 'percent'
                ylabel = "Percentage (%)"
            else:
                day_counts = ((clst_df['Dia_da_semana'].value_counts())).sort_index()
                stat = 'count'
                ylabel = "No. of observations"
            #day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index() 
            print((day_counts))
            
            plt.figure(figsize=(10,5))
            ax = sns.histplot(data=clst_df, x="Dia_da_semana", stat=stat, discrete=True, shrink=.8)
            plt.title("Cluster " + str(c))
            plt.xlabel("Day of Week")
            plt.xticks(range(7), week_days, rotation='horizontal')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False)
            
            plt.ylabel(ylabel)
            if mode == 'perc':
                plt.ylim(0, 100)

            #plt.show()
            plt.savefig(save+'day_of_week_cluster'+str(c)+'.png', dpi=250)

    # Show clusters distribution by month
    if approach not in ['HWDC']:
        # Months
        month_dic = {
            1:  'Jan',
            2:  'Feb',
            3:  'Mar',
            4:  'Apr',
            5:  'May',
            6:  'Jun',
            7:  'Jul',
            8:  'Ago',
            9:  'Sep',
            10: 'Oct',
            11: 'Nov',
            12: 'Dec',
        }
        df_aux['Month'] = df_aux['Mes'].replace(month_dic)

        df_aux2 = df_aux.copy()
        df_aux2['Mes'].replace(month_dic,inplace=True)

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
        for c in range(n_clusters):
            clst_df = df_aux2[df_aux2['Cluster'] == c]
            print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
            if mode == 'perc':
                day_counts = ((clst_df['Mes'].value_counts(normalize=True))*100).sort_index()
                stat = 'percent'
                ylabel = "Percentage (%)"
            else:
                day_counts = ((clst_df['Mes'].value_counts())).sort_index()
                stat = 'count'
                ylabel = "No. of observations"
            
            #day_counts = ((clst_df['Mes'].value_counts(normalize=True))*100).sort_index() 
            print((day_counts))
            
            plt.figure(figsize=(10,5))
            ax = sns.histplot(data=clst_df, x="Month", stat=stat, discrete=True, shrink=.8)
            plt.title("Cluster " + str(c), fontsize=16)
            plt.xlabel("Month", fontsize=14)
            plt.xticks(range(12), months, rotation='horizontal')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False)
            plt.ylabel(ylabel, fontsize=14)
            plt.ylim(0, 100)
            #plt.show()
            plt.savefig(save+'month_cluster'+str(c)+'.png', dpi=250)


#%%
# DAY OF WEEK, MONTH, SEASON DISTRIBUTION
################################################################################

def uniquify(path):
    '''
    If file exists create a new file with a (number) as a sufix
    '''
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


# Adds column 'Season' taking in account the year day of each row (without sorting)
def day2season(df):
    new_df = df.copy(deep=True)
    for season in ['spring', 'summer', 'autumn', 'winter']:
        if season == 'spring':
            new_df.loc[(new_df['Dia'] >= 79) & (new_df['Dia'] <= 171), 'Season'] = 0
        elif season == 'summer':
            new_df.loc[(new_df['Dia'] >= 172) & (new_df['Dia'] <= 264), 'Season'] = 1
        elif season == 'autumn':
            new_df.loc[(new_df['Dia'] >= 265) & (new_df['Dia'] <= 354), 'Season'] = 2
        elif season == 'winter':
            new_df.loc[((new_df['Dia'] >= 355) & (new_df['Dia'] <= 365)) | ((new_df['Dia'] >= 0) & (new_df['Dia'] <= 78)), 'Season'] = 3

    return new_df


def wms_distribution(
        df, 
        ts,
        n_clusters,
        clust_func,
        dist_matrix=None,
        linkage_matrix=None,
        criterion='maxclust',
        t=2,
        df_A2=None,
        intervals=None,
        metric_params=None, 
        approach='HWDC', 
        mode='count',
        path='distribution/',
        clst_func_str='',
        save_dist_txt = True,
        ts_viz=False,
        ts_viz_conf=None,
        random_state=RANDOM_SEED,
        alpha=0.5,
        linewidth=0.6,
        text_x=1,
        text_y=0.22,
        text_font_size=11,
        colors=sns.color_palette().as_hex(),
        **kwargs
        ):
    """
    Performs distribution analysis and visualization of data using clustering.

    Parameters:
    -----------
    df : DataFrame, mandatory
        Full Data frame to get information about the data clustered (e.g. Local, Day of week, Month, ...).

    ts : Array, mandatory
        Data to be clustered.

    dist_matrix : Array (quadratic matrix), optional
        Pre-computed matrix of distances. Used in KMedoids when the metrics are DTW or Soft-DTW and there is the need to use method='precomputed'
    
    linkage_matrix: ndarray, optional
        [Only used in hierarchical clustering.]
        The hierarchical clustering encoded with the matrix returned by the linkage function.

    criterion: str, optional
        [Only used in hierarchical clustering.]
        The criterion to use in forming flat clusters. This can be any of the following values: 'inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit'.
        
    t: int, optional
        [Only used in hierarchical clustering.]
        For criteria 'inconsistent', 'distance' or 'monocrit', this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria, this would be max number of clusters requested.

    n_clusters : int
        The number of clusters the data is to be split into.

    clust_func :
        Clustering function to be used (e.g. TimeSeriesKMeans, KMedoids, KMeans).

    df_A2 : DataFrame, optional
        If a DataFrame is to be clustered (Approach 2 or CV) it should be placed here.

    intervals : List[int], optional
        List of the hours that define the time intervals in which Approach 2 DataFrame was created.
        (Only if an Approach 2 DataFrame is to be clustered.)

    metric_params : dict, optinal
        Parameter values for the chosen metric. Used when clust_func is TimeSeriesKMeans and you want to add a constraint like Sakoe Chiba (e.g. {'sakoe_chiba_radius': 2}).

    approach : str, mandatory
        Name of the aggregation approach (e.g. 'HWDC', 'CV', ...)

    mode : str, optional
        The way the graphics show the distribution of the clustered data:
            - number of observations ('count');
            - percentage ('perc').

    path : str, optional
        The path to the directory where the gaphics are to be saved

    clst_func_str : str, optional
        The identification of the clustering function or metric used so, when saving the graphics it can be written in their name.

    save_dist_txt : bool, optional
        If True saves the distribution values of the clusters in a txt file.

    ts_viz : bool, optional
        If True shows the graphic with all time series for each cluster and its barycenter and average 

    ts_viz_conf : dict, optional
        If ts_viz is True this dictionary sets the time series graphics with the following optinional keys:
            * 'average': 
            * 'barycenter': 
            * 'dynamic_limits': 
            * 'text': 
            * 'title': 
            * 'save': 
        If the 'CV' approach parameter is used, then the 'ts_viz_conf' function only recognizes the 'save' key if it is explicitly specified.

    random_state : int, optional
        Generator used to initialize the centers. If an integer is given, it fixes the seed.

    **kwargs : keyword parameters, optional
        Any further parameters are passed directly to the clustering function.  

    Returns:
    --------
    Clustering elapsed time (str)

    """
    
    # Adjustments on df for the respective aggregation approach
    if approach == 'GDC':
        df_aux = df.drop_duplicates(subset=['Dia'])[['Dia','Dia_da_semana','Mes']]
        df_aux = df_aux.sort_values(by=['Dia'])

    elif (approach == 'HDC') or (approach == 'CV'):
        df_aux = df.drop_duplicates(subset=['Local'])[['Local']]
        df_aux = df_aux.sort_values(by=['Local'])

    elif approach == 'HMDC':
        df_aux = df.sort_values(by=['Local', 'Mes'])
        df_aux.drop_duplicates(subset=['Local', 'Mes'], inplace=True)
        df_aux = df_aux[['Dia','Mes']]

    elif approach == 'HWDC':
        df_aux = df.sort_values(by=['Local', 'Dia_da_semana'])
        df_aux.drop_duplicates(subset=['Local', 'Dia_da_semana'], inplace=True)
        df_aux = df_aux[['Dia','Dia_da_semana']]

    elif approach == 'HSDC':
        df_aux = day2season(df)
        df_aux = df_aux.sort_values(by=['Local', 'Season'])
        df_aux.drop_duplicates(subset=['Local', 'Season'], inplace=True)
        df_aux = df_aux[['Dia']]

    elif approach == 'HWMDC':
        df_aux = df.sort_values(by=['Local', 'Mes', 'Dia_da_semana'])
        df_aux.drop_duplicates(subset=['Local', 'Mes', 'Dia_da_semana'], inplace=True)
        df_aux = df_aux[['Dia', 'Mes', 'Dia_da_semana']]

    elif approach == 'DC':
        df_aux = df.sort_values(by=['Local', 'Dia'])
        df_aux.drop_duplicates(subset=['Local', 'Dia'], inplace=True)
        df_aux = df_aux[['Dia', 'Mes', 'Dia_da_semana']]


    # Run the clustering algorithm
    if clust_func == TimeSeriesKMeans:
        start = datetime.datetime.now()

        km = clust_func(n_clusters=n_clusters, init='k-means++', random_state=random_state, metric_params=metric_params, **kwargs)
        km.fit(ts)
        labels = km.predict(ts)

        stop = datetime.datetime.now()
        elapsed_time = stop - start
        print(f"time elapsed: {elapsed_time}")

    elif clust_func == KMeans:
        start = datetime.datetime.now()

        km = clust_func(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state, **kwargs)
        
        if df_A2 is not None:
            print(df_A2.info())
            km.fit(df_A2)
            labels = km.predict(df_A2)
        else:
            km.fit(ts)
            labels = km.predict(ts)

        stop = datetime.datetime.now()
        elapsed_time = stop - start
        print(f"time elapsed: {elapsed_time}")

    elif clust_func == KMedoids:
        start = datetime.datetime.now()

        km = clust_func(n_clusters=n_clusters, init='k-medoids++', random_state=random_state, **kwargs)
        
        if df_A2 is not None:
            km.fit(df_A2)
            labels = km.predict(df_A2)
        elif dist_matrix is not None:
            km.fit(dist_matrix)
            labels = km.predict(dist_matrix)
        else:
            km.fit(ts)
            labels = km.predict(ts)

        stop = datetime.datetime.now()
        elapsed_time = stop - start
        print(f"time elapsed: {elapsed_time}")

    # Hierarchical clustering
    elif clust_func == fcluster:
        start = datetime.datetime.now()
        labels = fcluster(linkage_matrix, t, criterion=criterion, **kwargs)
        stop = datetime.datetime.now()
        elapsed_time = stop - start
        print(f"time elapsed: {elapsed_time}")

        labels -= 1


    # Add cluster labels to the DataFrame
    df_aux['Cluster'] = labels.tolist()

    
    distrib_print = str(df_aux['Cluster'].value_counts())
    print(distrib_print)

    if save_dist_txt:
        full_unique_path = uniquify(path + approach + '_' + clst_func_str + '_dist.txt')

        # save global distribution in a txt file
        with open(full_unique_path,"a+") as f:
                f.write(distrib_print + '\n')


    # DISTRIBUTION BY DAY OF WEEK #
    # IF NOT HMDC or HDC - Show clusters distribution by day of week
    # not in ['HMDC', 'HDC', 'CV']
    if approach in ['GDC', 'HWDC', 'HWMDC', 'DC']:
        w_distrib_print = f"\nDISTRIBUTION BY DAY OF WEEK\n"

        # Days of week => Sun = 0, Sat = 6
        week_days_en = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        week_days = ['Dom', 'Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb']

        print(f'\nDay of Week:')
        for c in range(n_clusters):
            clst_df = (df_aux[df_aux['Cluster'] == c])

            # confirmação da distribuição
            print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
            print((clst_df['Dia_da_semana'].value_counts()).sort_index())
            print((clst_df['Dia_da_semana'].value_counts()))
            
            if mode == 'perc':
                day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index()
                stat = 'percent'
                ylabel = "Percentagem (%)"
            else:
                day_counts = ((clst_df['Dia_da_semana'].value_counts())).sort_index()
                stat = 'count'
                ylabel = "Número de observações"
            #day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index() 
            
            w_distrib_print += f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)\n" \
                             + f"{day_counts}\n\n"

            # Histogram            
            plt.figure(figsize=(10,5))
            ax = sns.histplot(data=clst_df, x="Dia_da_semana", stat=stat, discrete=True, shrink=.8)
            plt.title("Cluster " + str(c))
            plt.xlabel("Dia da semana")
            plt.xticks(range(7), week_days, rotation='horizontal')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False)
            
            plt.ylabel(ylabel)
            if mode == 'perc':
                plt.ylim(0, 100)

            #plt.show()
            plt.savefig(path+approach+'_'+clst_func_str+'_DoWeek_cluster_'+str(c)+'.png', dpi=250)

        if save_dist_txt:
            with open(full_unique_path,"a+") as f:
                f.write(w_distrib_print + '\n')
        else:
            print(w_distrib_print)


    # DISTRIBUTION BY MONTH #
    # IF NOT HWDC or HDC - Show clusters distribution by month
    # not in ['HWDC', 'HDC', 'CV']
    if approach in ['GDC', 'HMDC', 'HWMDC', 'DC']:
        m_distrib_print = f"\nDISTRIBUTION BY MONTH\n"

        # Months
        month_dic_en = {
            1:  'Jan',
            2:  'Feb',
            3:  'Mar',
            4:  'Apr',
            5:  'May',
            6:  'Jun',
            7:  'Jul',
            8:  'Ago',
            9:  'Sep',
            10: 'Oct',
            11: 'Nov',
            12: 'Dec',
        }

        # Months (PT)
        month_dic = {
            1:  'Jan',
            2:  'Fev',
            3:  'Mar',
            4:  'Abr',
            5:  'Mai',
            6:  'Jun',
            7:  'Jul',
            8:  'Ago',
            9:  'Set',
            10: 'Out',
            11: 'Nov',
            12: 'Dez',
        }

        # confirmação da distribuição
        print("\nMonth:")
        for c in range(n_clusters):
            clst_df = df_aux[df_aux['Cluster'] == c]
            print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
            print((clst_df['Mes'].value_counts()).sort_index())
            print((clst_df['Mes'].value_counts()))

        #df_aux['Month'] = df_aux['Mes'].replace(month_dic)

        #df_aux2 = df_aux.copy()
        #df_aux2['Mes'].replace(month_dic,inplace=True)

        months_en = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        for c in range(n_clusters):
            clst_df = df_aux[df_aux['Cluster'] == c]
            #print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
            if mode == 'perc':
                day_counts = ((clst_df['Mes'].value_counts(normalize=True))*100).sort_index()
                stat = 'percent'
                ylabel = "Percentagem (%)"
            else:
                day_counts = ((clst_df['Mes'].value_counts())).sort_index()
                stat = 'count'
                ylabel = "Número de observações"
            
            #day_counts = ((clst_df['Mes'].value_counts(normalize=True))*100).sort_index() 
            #print((day_counts))
            m_distrib_print += f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)\n" \
                             + f"{day_counts}\n\n"
            
            plt.figure(figsize=(10,5))
            ax = sns.histplot(data=clst_df, x="Mes", stat=stat, discrete=True, shrink=.8)
            plt.title("Cluster " + str(c), fontsize=16)
            plt.xlabel("Mês", fontsize=14)
            plt.xticks(range(1,13), months, rotation='horizontal')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False)
            plt.ylabel(ylabel, fontsize=14)
            if mode == 'perc':
                plt.ylim(0, 100)
            #plt.show()
            plt.savefig(path+approach+'_'+clst_func_str+'_Month_cluster_'+str(c)+'.png', dpi=250)

        if save_dist_txt:
            with open(full_unique_path,"a+") as f:
                f.write(m_distrib_print + '\n')
        else:
            print(m_distrib_print)


    # DISTRIBUTION BY SEASON #
    if approach in ['GDC', 'HMDC', 'HWMDC', 'HSDC', 'DC']:
        s_distrib_print = f"\nDISTRIBUTION BY SEASON\n"

        df_aux = day2season(df_aux)

        seasons_dic = {
            0:  'Primavera',
            1:  'Verão',
            2:  'Outono',
            3:  'Inverno',
        }

        # confirmação da distribuição
        print("\nSeason:")
        for c in range(n_clusters):
            clst_df = df_aux[df_aux['Cluster'] == c]
            print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
            print(clst_df['Season'].value_counts().sort_index())
            print(clst_df['Season'].value_counts())

        #df_aux['Estacao'] = df_aux['Season'].replace(seasons_dic)
        #df_aux2 = df_aux.copy()
        #df_aux2['Season'].replace(seasons_dic,inplace=True)


        seasons_en = ['Spring', 'Summer', 'Autumn', 'Winter']
        seasons = ['Primavera', 'Verão', 'Outono', 'Inverno']

        for c in range(n_clusters):
            clst_df = df_aux[df_aux['Cluster'] == c]

            if mode == 'perc':
                day_counts = ((clst_df['Season'].value_counts(normalize=True))*100).sort_index()
                stat = 'percent'
                ylabel = "Percentagem (%)"
            else:
                day_counts = ((clst_df['Season'].value_counts())).sort_index()
                stat = 'count'
                ylabel = "Número de observações"

            s_distrib_print += f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)\n" \
                             + f"{day_counts}\n\n"
            
            plt.figure(figsize=(10,5))
            ax = sns.histplot(data=clst_df, x="Season", stat=stat, discrete=True, shrink=.8)
            plt.title("Cluster " + str(c), fontsize=16)
            plt.xlabel("Estação", fontsize=14)
            plt.xticks(range(4), seasons, rotation='horizontal')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False)
            plt.ylabel(ylabel, fontsize=14)
            if mode == 'perc':
                plt.ylim(0, 100)
            #plt.show()
            plt.savefig(path+approach+'_'+clst_func_str+'_Season_cluster_'+str(c)+'.png', dpi=250)

        if save_dist_txt:
            with open(full_unique_path,"a+") as f:
                f.write(s_distrib_print + '\n')
        else:
            print(s_distrib_print)


    # Show visualizations as barycenter, average, all time series for each cluster 
    if ts_viz:
        unique, counts = np.unique(labels, return_counts=True)
        #print("Distribution of cluster elements => ", str(dict(zip(unique, counts))))

        # Centroids
        # Cluster centroids are calculated by taking the mean of daily
        # profiles that belongs to the assigned cluster number
        average_centroid=[]
        for k in range(n_clusters):
            #indices = np.where(labels == k)
            #a1_clust_match = ts[indices]
            a1_clust_match = ts[labels == k]
            average_centroid.append(sum(a1_clust_match)/len(a1_clust_match))
        #print('AVERAGE_CENTROIDS:')
        #print(average_centroid)

        total_series = len(ts)
        
        for yi in range(n_clusters):
            plt.figure()
            #plt.subplot(n_clusters, 1, yi + 1)
            series = 0
            color = colors[yi]
            for xx in ts[labels == yi]:
                plt.plot(xx.ravel(), color, alpha=alpha, linewidth=linewidth)
                series += 1
            
            if ('average' in ts_viz_conf) and (ts_viz_conf['average'] == True):
                plt.plot(average_centroid[yi], "red", linewidth=2, label='média')
            if ('barycenter' in ts_viz_conf) and (ts_viz_conf['barycenter'] == True):
                # barycenter is the average when comes to KMedoids algorithm
                if clust_func == KMedoids and ('metric' in kwargs and kwargs.get("metric") == 'precomputed'):
                    #plt.plot(average_centroid[yi], "orange", linewidth=2, label='centróide')
                    plt.plot(ts[km.medoid_indices_[yi]], "orange", linewidth=2, label='medoid')
                    
                else:
                    centroids = km.cluster_centers_
                    # TODO: Corrigir de forma a apresentar os centroids no A2
                    
                    if df_A2 is not None:
                        """
                        A2_centroids = centroids[yi].ravel()
                        centroids_ = [None] * 24
                        j = 0
                        for i in range(len(intervals)-1):
                            init = intervals[i]
                            end = intervals[i+1]
                            while init != end:
                                centroids_[init] = A2_centroids[j]
                                j+=1
                                init+=1
                        """
                        centroids_ = centroids[yi].ravel()
                    else:
                        centroids_ = centroids[yi].ravel()

                    if clust_func == KMedoids:
                        plt.plot(centroids_, "orange", linewidth=2, label='medoid')
                    else:
                        plt.plot(centroids_, "orange", linewidth=2, label='centróide')
            
                if ('barycenter_metric' in ts_viz_conf):
                    ts_cluster = ts[labels == yi]
                    barycenter = np.zeros(24)
                    
                    if (ts_viz_conf['barycenter_metric'] == 'euclidean'):
                        barycenter = euclidean_barycenter(ts_cluster)
                    elif (ts_viz_conf['barycenter_metric'] == 'dtw'):
                        barycenter = dtw_barycenter_averaging(ts_cluster)
                    elif (ts_viz_conf['barycenter_metric'] == 'dtw_sakoe2'):
                        barycenter = dtw_barycenter_averaging(ts_cluster, metric_params={'sakoe_chiba_radius': 2})
                    elif (ts_viz_conf['barycenter_metric'] == 'soft_dtw'):
                        barycenter = softdtw_barycenter(ts_cluster)

                    plt.plot(barycenter, "blue", linewidth=2, label='barycenter')

            plt.xlim(0, ts.shape[1])

            if ('dynamic_limits' in ts_viz_conf) and (ts_viz_conf['dynamic_limits'] == True):
                plt.ylim(-0.01,ts[labels == yi].max()*1.2)
            else:
                plt.ylim(-0.01,ts.max()*1.2)
            plt.xticks(range(0, 24, 2))

            if ('text' in ts_viz_conf) and (ts_viz_conf['text'] == True):
                txt = plt.text(text_x, text_y, f'{series} séries temporais', fontsize = text_font_size)
                txt.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
            plt.legend()

            if 'title' in ts_viz_conf:
                plt.title(ts_viz_conf['title'] + " - Cluster " + str(yi))
            else:
                plt.title("Cluster " + str(yi))

            if ('save' in ts_viz_conf) and (ts_viz_conf['save'] == True):
                full_unique_path_png = uniquify(path + approach + '_' + clst_func_str + '_cluster_'+ str(yi) + '.png')
                plt.savefig(full_unique_path_png, dpi=250)

    # 2D visualization
    if approach == 'CV':
        plt.figure(figsize=(14,7))
        sns.scatterplot(
            x=df_A2.columns[0],y=df_A2.columns[1],
            hue=labels,
            palette=sns.color_palette("hls",10),
            data=df_A2, 
            legend="full")
        if ('save' in ts_viz_conf) and (ts_viz_conf['save'] == True):
                full_unique_path_png = uniquify(path + 'CV_2D.png')
                plt.savefig(full_unique_path_png, dpi=250)
        else:
            plt.show()

    plt.close('all')

    return elapsed_time, labels



#######################################################
#            Visualization with PCA                   #
#######################################################
from sklearn.decomposition import PCA  # Principal Component Analysis

def clusters_2d_viz(X, n_clusters, save=''):
    #plotX is a DataFrame containing 5000 values sampled randomly from X
    #X = dfr.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    clusters = kmeans.fit_predict(X)
    X["Cluster"] = clusters
    #plotX = pd.DataFrame(np.array(X.sample(5000))) #subsample
    plotX = X

    # Rename plotX's columns since it was briefly converted to an np.array above
    plotX.columns = X.columns

    # PCA with two principal components
    pca_2d = PCA(n_components=2)

    # This DataFrame contains the two principal components that will be used
    # for the 2-D visualization mentioned above
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

    # "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    plotX = pd.concat([plotX,PCs_2d], axis=1, join='inner')

    #Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
    #This is because we intend to plot the values contained within each of these DataFrames.

    traces = []
    for c in range(n_clusters):
        cluster = plotX[plotX["Cluster"] == c]
        traces.append(go.Scatter(
                                x = cluster["PC1_2d"], 
                                y = cluster["PC2_2d"],
                                mode = "markers",
                                name = "Cluster " + str(c),
                                #marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                                text = None))

    title = "Clusters 2D Visualization (using PCA)"
    
    '''
    title={
        'text': "Clusters 2D Visualization (using PCA)",
        # 'y':0.9,
        # 'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    '''

    layout = dict(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                )

    fig = dict(data = traces, layout = layout)

    iplot(fig)

    if save != '':
        plt.savefig(save, dpi=250)

# %%
