#%%
import pickle
import os
from utils_optimal_k import *
import datetime
from numpy import load


clusters_k = range(2,51)


#                 NORMALIZED DATA                  #
####################################################

# Load the time-series and dataframe objects:
with open('Data/ts_df_norm.pkl', 'rb') as f:
    ts_gdc_norm, df_gdc_norm, ts_hdc_norm, df_hdc_norm, ts_hmdc_norm, df_hmdc_norm, ts_hwdc_norm, df_hwdc_norm = pickle.load(f)

with open('Data/ts_df_norm2.pkl', 'rb') as f:
    ts_dc_norm, df_dc_norm, ts_hsdc_norm, df_hsdc_norm, ts_hwmdc_norm, df_hwmdc_norm = pickle.load(f)

agg_app_norm = [
    ('GDC', ts_gdc_norm, df_gdc_norm),       # 0
    ('HDC', ts_hdc_norm, df_hdc_norm),       # 1
    ('HMDC', ts_hmdc_norm, df_hmdc_norm),    # 2
    ('HWDC', ts_hwdc_norm, df_hwdc_norm),    # 3
    ('HWMDC', ts_hwmdc_norm, df_hwmdc_norm), # 4
    ('HSDC', ts_hsdc_norm, df_hsdc_norm),    # 5
    ('DC', ts_dc_norm, df_dc_norm)           # 6
    ]

#%%
import sklearn 
print('sklearn: {}'. format(sklearn. __version__))

#%%
i=0

agg_app_norm[i][1]

#%%
# APROACH 2 #
print("\n\n######### APROACH 2 #########")


#for i in [4,6]:
#for i in range(7):
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")

    if i in [0,1,2,3,5]:
        ap_2 = agg_app_norm[i][2]
    elif i in [4,6]:
        with open('Data/df_perm_'+agg_app_norm[i][0]+'_norm.pkl', 'rb') as f:
            ap_2 = pickle.load(f)

    print("\nKMEANS")
    savehere = 'results/NORM/Approach_2/KMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl'

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
    """

    print("\n## SILHOUETTE METHOD EUCLIDEAN w/ KMeans:")
    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap_2,
        clusters_k,
        clust_func=KMeans,
        init='k-means++',
        sil_metric="euclidean",
        A2=True,
        ts=agg_app_norm[i][1]
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    

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
    print("\nKMEDOIDS")
    savehere = 'results/NORM/Approach_2/KMedoids/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl'

    
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
    

    print("\n## Calinski-Harabasz Index METHOD EUCLIDEAN w/ KMedoids:")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
        ap_2,
        clusters_k,
        KMedoids,
        metric="euclidean",
        save=savehere+'__harabasz',
        metric_yb='calinski_harabasz'
    )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'__harabasz.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
    """

# %%
