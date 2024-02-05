#%%
import pickle
import os
from utils_optimal_k import *
import datetime
from numpy import load


clusters_k = range(2,51)

i = 3


#%%
#                     RAW DATA                     #
####################################################

# Load the time-series and dataframe objects:
with open('Data/ts_df.pkl', 'rb') as f:
    ts_gdc, df_gdc, ts_hdc, df_hdc, ts_hmdc, df_hmdc, ts_hwdc, df_hwdc = pickle.load(f)

with open('Data/ts_df_2.pkl', 'rb') as f:
    ts_dc, df_dc, ts_hsdc, df_hsdc, ts_hwmdc, df_hwmdc = pickle.load(f)


agg_app_raw = [
    ('GDC', ts_gdc, df_gdc, 2005.2571047501792),       # 0
    ('HDC', ts_hdc, df_hdc, 1911.7009796191983),       # 1
    ('HMDC', ts_hmdc, df_hmdc, 2038.0770031217473),    # 2
    ('HWDC', ts_hwdc, df_hwdc, 3019.3766319963424),    # 3
    ('HWMDC', ts_hwmdc, df_hwmdc, 1505.2799999999995), # 4
    ('HSDC', ts_hsdc, df_hsdc, 3950.0913722513587),    # 5
    ('DC', ts_dc, df_dc, 431.99999999999994)           # 6
    ]

#%%
#############
# APROACH 1 #
#############

ap_1_ts = agg_app_raw[i][1]

if i==4:
    with open('Data/ts_perm_HWMDC_raw.pkl', 'rb') as f:
        ap_1_ts = pickle.load(f)

    ap_1_ts = ap_1_ts.reshape(10000,24,1)
elif i==6:
    with open('Data/ts_perm_DC_raw.pkl', 'rb') as f:
        ap_1_ts = pickle.load(f)
        
    ap_1_ts = ap_1_ts.reshape(10000,24,1)

ap_1 = ap_1_ts.squeeze()

path = 'tests/dist_mat/'

print(f"\n\n########   {agg_app_raw[i][0]}  ########")


#%%
# EUCLIDEAN
##########################
savehere = 'results/RAW/Approach_1/TSKMeans/'+agg_app_raw[i][0]+'/'+agg_app_raw[i][0]+'_eucl'

"""
print("\n\nSILHOUETTE EUCLIDEAN\n")

start = datetime.datetime.now()
avg_silhouettes, inertia, n_iter = silhouette(
    ap_1_ts,
    clusters_k, 
    TimeSeriesKMeans,
    sil_metric="euclidean",
    metric="euclidean",
    n_jobs=-1
    )
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

#%%
print("\nELBOW EUCLIDEAN\n")
start = datetime.datetime.now()
k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_1, 
        clusters_k,
        TimeSeriesKMeans,
        metric="euclidean",
        save=savehere+'_elbow'
        )
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'_elbow.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))

"""
#%%
print("\nCALINSKY HARABASZ - EUCLIDEAN\n")
start = datetime.datetime.now()
k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_1, 
        clusters_k,
        TimeSeriesKMeans,
        metric="euclidean",
        save=savehere+'_harabasz',
        metric_yb='calinski_harabasz'
        )
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'_harabasz.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))



#%%

# DTW
##########################
dist_matrix = load(path+'dist_mat_dtw_'+agg_app_raw[i][0]+'_raw.npy')
savehere = 'results/RAW/Approach_1/TSKMeans/'+agg_app_raw[i][0]+'/'+agg_app_raw[i][0]+'_dtw'

"""
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

"""

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


#%%
# DTW w/ Sakoe Chiba r=2
##########################
dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_raw[i][0]+'_raw.npy')
savehere = 'results/RAW/Approach_1/TSKMeans/'+agg_app_raw[i][0]+'/'+agg_app_raw[i][0]+'_dtw_sakoe_2'

"""
print("\n\nSILHOUETTE DTW with Sakoe Chiba r=2\n")

start = datetime.datetime.now()
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
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

#%%
print("\nELBOW DTW with Sakoe Chiba r=2\n")
start = datetime.datetime.now()
k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_1, 
        clusters_k,
        TimeSeriesKMeans,
        metric="dtw",
        metric_params={'sakoe_chiba_radius': 2},
        save=savehere+'_elbow'
        )
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'_elbow.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))

"""

#%%
print("\nCALINSKY HARABASZ - DTW with Sakoe Chiba r=2\n")
start = datetime.datetime.now()
k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_1, 
        clusters_k,
        TimeSeriesKMeans,
        metric="dtw",
        metric_params={'sakoe_chiba_radius': 2},
        save=savehere+'_harabasz',
        metric_yb='calinski_harabasz'
        )
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'_harabasz.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))


#%%
# Soft-DTW
##########################
dist_matrix = load(path+'dist_mat_soft_dtw_'+agg_app_raw[i][0]+'_raw.npy')
savehere = 'results/RAW/Approach_1/TSKMeans/'+agg_app_raw[i][0]+'/'+agg_app_raw[i][0]+'_soft_dtw'

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
    metric_params={"gamma": agg_app_raw[i][2]},
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
        metric_params={"gamma": agg_app_raw[i][2]},
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
        metric_params={"gamma": agg_app_raw[i][2]},
        save=savehere+'_harabasz',
        metric_yb='calinski_harabasz'
        )
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere+'_harabasz.npy', 'wb') as f:
    np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
    
