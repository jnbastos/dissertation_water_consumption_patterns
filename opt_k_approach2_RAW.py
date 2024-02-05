#%%
import pickle
import os
from utils_optimal_k import *
import datetime
from numpy import load


clusters_k = range(2,51)


#%%
#                     RAW DATA                     #
####################################################

# Load the time-series and dataframe objects:
with open('Data/ts_df.pkl', 'rb') as f:
    ts_gdc, df_gdc, ts_hdc, df_hdc, ts_hmdc, df_hmdc, ts_hwdc, df_hwdc = pickle.load(f)

with open('Data/ts_df_2.pkl', 'rb') as f:
    ts_dc, df_dc, ts_hsdc, df_hsdc, ts_hwmdc, df_hwmdc = pickle.load(f)


agg_app_raw = [
    ('GDC', ts_gdc, df_gdc),       # 0
    ('HDC', ts_hdc, df_hdc),       # 1
    ('HMDC', ts_hmdc, df_hmdc),    # 2
    ('HWDC', ts_hwdc, df_hwdc),    # 3
    ('HWMDC', ts_hwmdc, df_hwmdc), # 4
    ('HSDC', ts_hsdc, df_hsdc),    # 5
    ('DC', ts_dc, df_dc)           # 6
    ]



#%%
# APROACH 2 #
print("\n\n######### APROACH 2 - RAW #########")

#for i in [0,1,2,3,5]:
for i in [1]:
    print(f"\n\n########   {agg_app_raw[i][0]}  ########")

    if i in [0,1,2,3,5]:
        ap_2 = agg_app_raw[i][2]
    elif i in [4,6]:
        with open('Data/df_perm_'+agg_app_raw[i][0]+'_raw.pkl', 'rb') as f:
            ap_2 = pickle.load(f)

    # K-MEANS #
    print("\nKMEANS")
    savehere = 'results/RAW/Approach_2/KMeans/'+agg_app_raw[i][0]+'/'+agg_app_raw[i][0]+'_eucl'

    """
    print("\n## ELBOW METHOD EUCLIDEAN w/ KMeans:")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_2,
        clusters_k,
        KMeans,
        init='k-means++',
        save=savehere+'_elbow'
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_elbow.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
    

    print("\n## SILHOUETTE METHOD EUCLIDEAN w/ KMeans:")
    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap_2,
        clusters_k,
        KMeans,
        init='k-means++',
        sil_metric="euclidean"
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    """

    print("\n## Calinski-Harabasz Index METHOD EUCLIDEAN w/ KMeans:")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_2,
        clusters_k,
        KMeans,
        init='k-means++',
        save=savehere+'_harabasz',
        metric_yb='calinski_harabasz'
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    print(k_scores)
    print(elbow_score)
    print(elbow_value)

    with open(savehere+'_harabasz.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
    


    """
    # K-MEDOIDS #
    print("\nKMEDOIDS")
    savehere = 'results/RAW/Approach_2/KMedoids/'+agg_app_raw[i][0]+'/'+agg_app_raw[i][0]+'_eucl'

    print("\n## ELBOW METHOD EUCLIDEAN w/ KMedoids:")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_2,
        clusters_k,
        KMedoids,
        metric="euclidean",
        save=savehere+'_elbow'
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_elbow.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))


    print("\n## SILHOUETTE METHOD EUCLIDEAN w/ KMedoids:")
    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap_2,
        clusters_k,
        KMedoids,
        sil_metric="euclidean",
        metric="euclidean",
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    """

# %%
