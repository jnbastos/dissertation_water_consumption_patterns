#%%
import pickle
import os
from utils_optimal_k import *
#import datetime
from numpy import load

from datetime import datetime
from datetime import timedelta
from datetime import time


clusters_k = range(2,51)

#i = 0


#                 NORMALIZED DATA                  #
####################################################

# Load the time-series and dataframe objects:
with open('Data/ts_df_norm.pkl', 'rb') as f:
    ts_gdc_norm, df_gdc_norm, ts_hdc_norm, df_hdc_norm, ts_hmdc_norm, df_hmdc_norm, ts_hwdc_norm, df_hwdc_norm = pickle.load(f)

with open('Data/ts_df_norm2.pkl', 'rb') as f:
    ts_dc_norm, df_dc_norm, ts_hsdc_norm, df_hsdc_norm, ts_hwmdc_norm, df_hwmdc_norm = pickle.load(f)

agg_app_norm = [
    ('GDC', ts_gdc_norm, df_gdc_norm, 0.022342086064455827),       # 0
    ('HDC', ts_hdc_norm, df_hdc_norm, 0.029046996788993626),       # 1
    ('HMDC', ts_hmdc_norm, df_hmdc_norm, 0.03548876835516948),     # 2
    ('HWDC', ts_hwdc_norm, df_hwdc_norm, 0.03942207089736752),     # 3
    ('HWMDC', ts_hwmdc_norm, df_hwmdc_norm, 0.041630791756188494), # 4
    ('HSDC', ts_hsdc_norm, df_hsdc_norm, 0.05450430616830972),     # 5
    ('DC', ts_dc_norm, df_dc_norm, 0.012899758129535067)           # 6
    ]

#%%
#############
# APROACH 1 #
#############
#for i in [0,1,2,3,5]:
for i in [3,5]:
    ap_1_ts = agg_app_norm[i][1]

    if i==4:
        with open('Data/ts_perm_HWMDC_norm.pkl', 'rb') as f:
            ap_1_ts = pickle.load(f)

        ap_1_ts = ap_1_ts.reshape(10000,24,1)
    elif i==6:
        with open('Data/ts_perm_DC_norm.pkl', 'rb') as f:
            ap_1_ts = pickle.load(f)
            
        ap_1_ts = ap_1_ts.reshape(10000,24,1)

    ap_1 = ap_1_ts.squeeze()

    path = 'tests/dist_mat/'

    print(f"\n\n########   {agg_app_norm[i][0]}  ########")


    # EUCLIDEAN
    ##########################
    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl'

    if i == 2:
        pass
    else:
        print("\n\nSILHOUETTE EUCLIDEAN\n")
        values_list = []
        for j in range(3):
            start = datetime.now()
            avg_silhouettes, inertia, n_iter = silhouette(
                ap_1_ts,
                clusters_k, 
                TimeSeriesKMeans,
                sil_metric="euclidean",
                metric="euclidean",
                n_jobs=-1
                )
            stop = datetime.now()
            elapsed_time = stop - start
            #print(f"time elapsed: {elapsed_time}")
            elapsed_time_ms = elapsed_time.total_seconds()*1000
            print(f"time elapsed (ms): {elapsed_time_ms}")

            values_list.append(elapsed_time_ms)

        arr = np.array(values_list)
        values_mean = np.mean(arr, axis=0)
        values_std = np.std(arr, axis=0)

        print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
        

        #with open(savehere+'.npy', 'wb') as f:
        #    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

        
        print("\nELBOW EUCLIDEAN\n")
        values_list = []
        for j in range(3):
            start = datetime.now()
            k_scores, k_timers, elbow_value, elbow_score = elbow(
                    ap_1, 
                    clusters_k,
                    TimeSeriesKMeans,
                    metric="euclidean",
                    save=savehere+'_elbow'
                    )
            stop = datetime.now()
            elapsed_time = stop - start
            #print(f"time elapsed: {elapsed_time}")
            elapsed_time_ms = elapsed_time.total_seconds()*1000
            print(f"time elapsed (ms): {elapsed_time_ms}")

            values_list.append(elapsed_time_ms)

        arr = np.array(values_list)
        values_mean = np.mean(arr, axis=0)
        values_std = np.std(arr, axis=0)

        print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
        
        #with open(savehere+'_elbow.npy', 'wb') as f:
        #    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))



        print("\nCALINSKY HARABASZ - EUCLIDEAN\n")
        values_list = []
        for j in range(3):
            start = datetime.now()
            k_scores, k_timers, elbow_value, elbow_score = elbow(
                    ap_1, 
                    clusters_k,
                    TimeSeriesKMeans,
                    metric="euclidean",
                    save=savehere+'_harabasz',
                    metric_yb='calinski_harabasz'
                    )
            stop = datetime.now()
            elapsed_time = stop - start
            #print(f"time elapsed: {elapsed_time}")
            elapsed_time_ms = elapsed_time.total_seconds()*1000
            print(f"time elapsed (ms): {elapsed_time_ms}")

            values_list.append(elapsed_time_ms)

        arr = np.array(values_list)
        values_mean = np.mean(arr, axis=0)
        values_std = np.std(arr, axis=0)

        print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
        

        #with open(savehere+'_harabasz.npy', 'wb') as f:
        #    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))



    """
    # DTW
    ##########################
    dist_matrix = load(path+'dist_mat_dtw_'+agg_app_norm[i][0]+'_norm.npy')
    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_dtw'


    print("\n\nSILHOUETTE DTW\n")

    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap_1_ts,
        clusters_k, 
        TimeSeriesKMeans,
        dist_matrix=dist_matrix,
        sil_metric="precomputed",
        metric="dtw",
        n_jobs=-1
        )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    #%%
    with open(savehere+'.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    #%%
    print("\nELBOW DTW\n")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
            ap_1, 
            clusters_k,
            TimeSeriesKMeans,
            metric="dtw",
            save=savehere+'_elbow'
            )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_elbow.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))


    #%%
    print("\nCALINSKY HARABASZ - DTW\n")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
            ap_1, 
            clusters_k,
            TimeSeriesKMeans,
            metric="dtw",
            save=savehere+'_harabasz',
            metric_yb='calinski_harabasz'
            )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_harabasz.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))

    """


    # DTW w/ Sakoe Chiba r=2
    ##########################
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_dtw_sakoe_2'

    if i == 2:
        pass
    else:
        print("\n\nSILHOUETTE DTW with Sakoe Chiba r=2\n")
        values_list = []
        for j in range(3):
            start = datetime.now()
            avg_silhouettes, inertia, n_iter = silhouette(
                ap_1_ts, 
                clusters_k, 
                TimeSeriesKMeans,
                dist_matrix=dist_matrix,
                sil_metric="precomputed",
                metric="dtw",
                metric_params={'sakoe_chiba_radius': 2},
                n_jobs=-1
                )
            stop = datetime.now()
            elapsed_time = stop - start
            #print(f"time elapsed: {elapsed_time}")
            elapsed_time_ms = elapsed_time.total_seconds()*1000
            print(f"time elapsed (ms): {elapsed_time_ms}")

            values_list.append(elapsed_time_ms)

        arr = np.array(values_list)
        values_mean = np.mean(arr, axis=0)
        values_std = np.std(arr, axis=0)

        print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
        

        #with open(savehere+'.npy', 'wb') as f:
        #    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

    #%%
    print("\nELBOW DTW with Sakoe Chiba r=2\n")
    values_list = []
    for j in range(3):
        start = datetime.now()
        k_scores, k_timers, elbow_value, elbow_score = elbow(
                ap_1, 
                clusters_k,
                TimeSeriesKMeans,
                metric="dtw",
                metric_params={'sakoe_chiba_radius': 2},
                save=savehere+'_elbow'
                )
        stop = datetime.now()
        elapsed_time = stop - start
        #print(f"time elapsed: {elapsed_time}")
        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)

    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
       

    #with open(savehere+'_elbow.npy', 'wb') as f:
    #    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))


    print("\nCALINSKY HARABASZ - DTW with Sakoe Chiba r=2\n")
    values_list = []
    for j in range(3):
        start = datetime.now()
        k_scores, k_timers, elbow_value, elbow_score = elbow(
                ap_1, 
                clusters_k,
                TimeSeriesKMeans,
                metric="dtw",
                metric_params={'sakoe_chiba_radius': 2},
                save=savehere+'_harabasz',
                metric_yb='calinski_harabasz'
                )
        stop = datetime.now()
        elapsed_time = stop - start
        #print(f"time elapsed: {elapsed_time}")
        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)

    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
     

    #with open(savehere+'_harabasz.npy', 'wb') as f:
    #    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))



    """
    # Soft-DTW
    ##########################
    dist_matrix = load(path+'dist_mat_soft_dtw_'+agg_app_norm[i][0]+'_norm.npy')
    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_soft_dtw'

    #%%
    print("\n\nSILHOUETTE Soft-DTW\n")

    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap_1_ts, 
        clusters_k, 
        TimeSeriesKMeans,
        dist_matrix=dist_matrix,
        sil_metric="precomputed",
        metric="softdtw",
        metric_params={"gamma": agg_app_norm[i][2]},
        n_jobs=-1
        )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    #%%
    print("\nELBOW Soft-DTW\n")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
            ap_1, 
            clusters_k,
            TimeSeriesKMeans,
            metric="softdtw",
            metric_params={"gamma": agg_app_norm[i][2]},
            save=savehere+'_elbow'
            )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_elbow.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
        

    #%%
    print("\nCALINSKY HARABASZ - Soft-DTW\n")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
            ap_1, 
            clusters_k,
            TimeSeriesKMeans,
            metric="softdtw",
            metric_params={"gamma": agg_app_norm[i][2]},
            save=savehere+'_harabasz',
            metric_yb='calinski_harabasz'
            )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_harabasz.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
        
    """