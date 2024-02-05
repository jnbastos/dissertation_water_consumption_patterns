#%%
import pickle
import os
from utils_optimal_k import *
import datetime
from numpy import load
import pandas as pd
from datetime import datetime
from datetime import time
from datetime import timedelta


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
# TIMES with k=2

# APPROACH 2 - KMEANS - Euclidean #
path = 'tests/dist_mat/'


for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    ap2 = agg_app_norm[i][2]

    a2KMeansEucl = []
    for j in range(5):
        start = datetime.now()
        km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=RANDOM_SEED)
        km.fit(ap2)
        stop = datetime.now()
        elapsed_time = str(stop - start)
        #print(f"time elapsed: {elapsed_time}")

        a2KMeansEucl.append('0' + elapsed_time)

    times = []
    for str_t in a2KMeansEucl:
        t = time.fromisoformat(str_t)
        times.append(timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond))

    df = pd.DataFrame({'a2KMeansEucl': times})
    print(df)
        
    df_ms = df.astype('timedelta64[ms]')
    print(df_ms)
    print(df_ms.describe())

#%%
# APPROACH 1 - KMEANS - Euclidean #
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    ap1 = agg_app_norm[i][1]

    a1KMeansEucl = []
    for j in range(5):
        start = datetime.now()
        km = TimeSeriesKMeans(n_clusters=2, init='k-means++', n_init=10, metric='euclidean', random_state=RANDOM_SEED)
        km.fit(ap1)
        stop = datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        a1KMeansEucl.append('0' + elapsed_time)

    times = []
    for str_t in a1KMeansEucl:
        t = time.fromisoformat(str_t)
        times.append(timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond))

    df = pd.DataFrame({'a1KMeansEucl': times})
    print(df)
        
    df_ms = df.astype('timedelta64[ms]')
    print(df_ms)
    print(df_ms.describe())

#%%
# APPROACH A1 - KMEDOIDS - DTW SC2 #
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
    ap1 = agg_app_norm[i][1]

    a1KMedoidsDTWSC2 = []
    for j in range(5):
        start = datetime.now()
        km = KMedoids(n_clusters=2, metric='precomputed', method='pam', init='k-medoids++', random_state=RANDOM_SEED)
        km.fit(dist_matrix)
        stop = datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        a1KMedoidsDTWSC2.append('0' + elapsed_time)

    times = []
    for str_t in a1KMedoidsDTWSC2:
        t = time.fromisoformat(str_t)
        times.append(timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond))

    df = pd.DataFrame({'a1KMedoidsDTWSC2': times})
    print(df)
        
    df_ms = df.astype('timedelta64[ms]')
    print(df_ms)
    print(df_ms.describe())

#%%



#%%


# %%
path = 'tests/dist_mat/'

# APPROACH 2 - KMEANS - EUCLIDEAN #
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    savehere = 'tests/silhouette_A2/'+agg_app_norm[i][0]+'_eucl'

    ap2 = agg_app_norm[i][2]

    labels = []
    for k in clusters_k:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_SEED)
        km.fit(ap2)
        labels.append(km.predict(ap2))

    #print(labels)

    print(f"\n## {agg_app_norm[i][0]}: Approach 2 - KMeans Euclidean - CH METHOD:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        labels_only=True,
        cluster_labels=labels,
        calinski=True
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_CH.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

    
    print(f"\n## {agg_app_norm[i][0]}: Approach 2 - KMeans Euclidean - SILHOUETTE METHOD w/DTW SC2")

    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        dist_matrix,
        clusters_k,
        KMeans,
        init='k-means++',
        sil_metric="precomputed",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_dtwsc2.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    

    print(f"\n## {agg_app_norm[i][0]}: Approach 2 - KMeans Euclidean - SILHOUETTE METHOD w/ Eucl:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        sil_metric="euclidean",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_eucl.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


# %%
path = 'tests/dist_mat/'

# APPROACH A2 - KMEDOIDS - DTW SC2 #
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    savehere = 'tests/silhouette_A2_kmedoids_dtwsc2/'+agg_app_norm[i][0]
    #dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
    dist_matrix = load(path+'dist_mat_A2_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    labels = []
    for k in clusters_k:
        km = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='k-medoids++', random_state=RANDOM_SEED)
        km.fit(dist_matrix)
        labels.append(km.predict(dist_matrix))

    #print(labels)

    print(f"\n## {agg_app_norm[i][0]}: Approach 2 - KMedoids DTW SC2 - CH METHOD:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        labels_only=True,
        cluster_labels=labels,
        calinski=True
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    #with open(savehere+'_CH.npy', 'wb') as f:
    #    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    print(f"\n## {agg_app_norm[i][0]}: Approach 2 KMedoids DTW SC2 - SILHOUETTE METHOD DTW SC2")
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        dist_matrix,
        clusters_k,
        sil_metric="precomputed",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    #with open(savehere+'_silhouette_dtwsc2.npy', 'wb') as f:
    #    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

    '''
    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids DTW SC2 - SILHOUETTE METHOD EUCLIDEAN:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        KMedoids,
        sil_metric="euclidean",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_eucl.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    '''


# %%
path = 'tests/dist_mat/'

# APPROACH A1 - KMEANS - EUCLIDEAN #
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    savehere = 'tests/silhouette_A1_kmeans_eucl/'+agg_app_norm[i][0]

    ap1 = agg_app_norm[i][1]

    labels = []
    for k in clusters_k:
        km = TimeSeriesKMeans(n_clusters=k, init='k-means++', n_init=10, metric='euclidean', random_state=RANDOM_SEED, n_jobs=-1)
        km.fit(ap1)
        labels.append(km.predict(ap1))

    #print(labels)

    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMeans euclidean - CH METHOD:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap1.squeeze(),
        clusters_k,
        labels_only=True,
        cluster_labels=labels,
        calinski=True
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_CH.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    
    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMeans euclidean - SILHOUETTE METHOD DTW SC2")

    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        dist_matrix,
        clusters_k,
        KMeans,
        init='k-means++',
        sil_metric="precomputed",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_dtwsc2.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    

    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMeans euclidean - SILHOUETTE METHOD EUCLIDEAN:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        init='k-means++',
        sil_metric="euclidean",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_eucl.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


# %%
path = 'tests/dist_mat/'

# APPROACH A1 - KMEDOIDS - DTW SC2 #
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    savehere = 'tests/silhouette_A1_kmedoids_dtwsc2/'+agg_app_norm[i][0]
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    ap1 = agg_app_norm[i][1]

    labels = []
    for k in clusters_k:
        km = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='k-medoids++', random_state=RANDOM_SEED)
        km.fit(dist_matrix)
        labels.append(km.predict(dist_matrix))

    #print(labels)

    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids DTW SC2 - CH METHOD:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap1.squeeze(),
        clusters_k,
        labels_only=True,
        cluster_labels=labels,
        calinski=True
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_CH.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids DTW SC2 - SILHOUETTE METHOD DTW SC2")
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        dist_matrix,
        clusters_k,
        sil_metric="precomputed",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_dtwsc2.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

    
    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids DTW SC2 - SILHOUETTE METHOD EUCLIDEAN:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        KMedoids,
        sil_metric="euclidean",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_eucl.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    

# %%
# APPROACH A1 - KMEANS - DTW SC r=2 #

path = 'tests/dist_mat/'

for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    savehere = 'tests/silhouette_A1_kmeans_dtwsc2/'+agg_app_norm[i][0]
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    ap1 = agg_app_norm[i][1]

    labels = []
    for k in clusters_k:
        km = TimeSeriesKMeans(n_clusters=k, init='k-means++', n_init=10, metric='dtw', metric_params={'sakoe_chiba_radius': 2}, random_state=RANDOM_SEED, n_jobs=-1)
        km.fit(ap1.squeeze())
        labels.append(km.predict(ap1.squeeze()))

    #print(labels)

    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMeans DTW SC2 - CH METHOD:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap1.squeeze(),
        clusters_k,
        TimeSeriesKMeans,
        labels_only=True,
        cluster_labels=labels,
        calinski=True
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_CH.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMeans DTW SC2 - SILHOUETTE METHOD DTW SC2")
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        dist_matrix,
        clusters_k,
        TimeSeriesKMeans,
        sil_metric="precomputed",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_dtwsc2.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))


    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMeans DTW SC2 - SILHOUETTE METHOD EUCLIDEAN:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        agg_app_norm[i][1].squeeze(),
        clusters_k,
        TimeSeriesKMeans,
        sil_metric="euclidean",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_eucl.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))



#%%
# APPROACH A1 - KMEDOIDS - Euclidean #
path = 'tests/dist_mat/'

for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    savehere = 'tests/silhouette_A1_kmedoids_eucl/'+agg_app_norm[i][0]
    dist_matrix = load(path+'dist_mat_eucl_'+agg_app_norm[i][0]+'_norm.npy')

    ap1 = agg_app_norm[i][1]

    labels = []
    for k in clusters_k:
        km = KMedoids(n_clusters=k, metric='euclidean', method='pam', init='k-medoids++', random_state=RANDOM_SEED)
        km.fit(ap1.squeeze())
        labels.append(km.predict(ap1.squeeze()))


    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids Euclidean - CH METHOD:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap1.squeeze(),
        clusters_k,
        KMedoids,
        labels_only=True,
        cluster_labels=labels,
        calinski=True
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    #with open(savehere+'_CH.npy', 'wb') as f:
    #    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

    """
    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids Euclidean - SILHOUETTE METHOD DTW SC2")
    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        dist_matrix,
        clusters_k,
        KMedoids,
        sil_metric="precomputed",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_silhouette_dtwsc2.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))
    """

    print(f"\n## {agg_app_norm[i][0]}: Approach 1 KMedoids Euclidean - SILHOUETTE METHOD EUCLIDEAN:")
    start = datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        ap1.squeeze(),
        clusters_k,
        KMedoids,
        sil_metric="euclidean",
        labels_only=True,
        cluster_labels=labels
    )
    stop = datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    #with open(savehere+'_silhouette_eucl.npy', 'wb') as f:
    #    np.save(f, np.array([elapsed_time, avg_silhouettes, inertia, n_iter], dtype=object))

#%%
# APPROACH A1 - HIERARCHICAL (Ward) - DTW SC r=2 #



#%%
# APPROACH A1 - HIERARCHICAL (Ward) - Euclidean #



#%%
dist_matrix1 = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
dist_matrix2 = load(path+'dist_mat_eucl_'+agg_app_norm[i][0]+'_norm.npy')

# %%
dist_matrix1.shape
# %%
dist_matrix2.shape
# %%
