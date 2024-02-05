#%%
from utils_clustering import *
import seaborn as sns; sns.set()  # for plot styling
import pickle
import numpy as np
from numpy import load
import pandas as pd



# %%
#                 NORMALIZED DATA                  #
####################################################

#df_norm = pd.read_csv(r'data/dfnor.csv')
df_norm_pkl = pd.read_pickle("Data/dfnor.pkl")

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
"""
Approach 1 (NORM DATA):
    * TSKMeans:
        - Euclidean
        - DTW
        - DTW Sakoe Chiba r=2
        (- Soft-DTW)

    * TSKMedoids:
        - Euclidean
        - DTW
        - DTW Sakoe Chiba r=2
        (- Soft-DTW)

    * (Hierarchical)

    
Approach 2 (NORM DATA):
    * KMeans:
        - Euclidean

    * KMedoids:
        - Euclidean

    * (Hierarchical)

    
CV (RAW DATA):
    * KMeans:
        - Euclidean

    * KMedoids:
        - Euclidean

    * (Hierarchical)

    
(Approach 1 (RAW DATA)) HDC & HWDC
(Approach 2 (RAW DATA)) HDC & HWDC
"""



ts_viz_conf = {
    'barycenter'    : True,
    'average'       : True,
    'dynamic_limits': True,
    'text'          : True,
    'save'          : True,
}

k = 3

for i in [0,1,2,3,5]:
    print(f"\n\n\n########   {agg_app_norm[i][0]}  ########")

    """
    # Approach 1 #
    ##############
    print("## Approach 1 ##")
    approach_path = 'distribution/NORM/Approach_1/'
    
    #* Time Series KMeans *#
    path = approach_path + 'TSKMeans/' + agg_app_norm[i][0] + '/'

    # TimeSeriesKMeans - Euclidean
    print("\n\n** TimeSeriesKMeans - Euclidean")

    ts_viz_conf['title'] = 'K-Means: Euclidean'

    values_list = []
    for j in range(5):
        elapsed_time = wms_distribution(
            df_norm_pkl,
            agg_app_norm[i][1],
            n_clusters=k,
            clust_func=TimeSeriesKMeans,
            metric="euclidean",
            approach=agg_app_norm[i][0],
            path=path,
            clst_func_str='euclidean_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )

        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)
    
    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
    """   


    """
    # TimeSeriesKMeans - DTW
    print("** TimeSeriesKMeans - DTW")

    ts_viz_conf['title'] = 'K-Means: DTW'

    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1],
        n_clusters=3,
        clust_func=TimeSeriesKMeans,
        metric="dtw",
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='dtw',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """
    
    """
    # TimeSeriesKMeans - DTW with Sakoe Chiba r=2
    print("\n** TimeSeriesKMeans - DTW with Sakoe Chiba r=2")

    ts_viz_conf['title'] = 'K-Means: DTW w/ Sakoe Chiba r=2'

    values_list = []
    for j in range(5):
        elapsed_time = wms_distribution(
            df_norm_pkl,
            agg_app_norm[i][1],
            n_clusters=k,
            clust_func=TimeSeriesKMeans,
            metric="dtw",
            metric_params={'sakoe_chiba_radius': 2},
            approach=agg_app_norm[i][0],
            path=path,
            clst_func_str='dtw_sakoe2_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )

        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)
    
    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
    """


    """
    # TimeSeriesKMeans - Soft-DTW
    print("** TimeSeriesKMeans - Soft-DTW")

    ts_viz_conf['title'] = 'K-Means: Soft-DTW'
    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1],
        n_clusters=3,
        clust_func=TimeSeriesKMeans,
        metric="softdtw",
        metric_params={"gamma": agg_app_norm[i][2]},
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='soft_dtw',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """

    """
    #* Time Series KMedoids *#
    path = approach_path + 'TSKMedoids/' + agg_app_norm[i][0] + '/'
    dist_matrix_path = 'tests/dist_mat/'

    # KMedoids - Euclidean
    print("\n\n** KMedoids - Euclidean")

    ts_viz_conf['title'] = 'K-Medoids: Euclidean'

    values_list = []
    for j in range(5):
        elapsed_time = wms_distribution(
            df_norm_pkl,
            agg_app_norm[i][1].squeeze(),
            n_clusters=k,
            clust_func=KMedoids,
            metric="euclidean",
            method='pam',
            approach=agg_app_norm[i][0],
            path=path,
            clst_func_str='euclidean_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )

        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)
    
    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
    """    


    """
    # KMedoids - DTW
    print("** KMedoids - DTW")

    dist_matrix = load(dist_matrix_path+'dist_mat_dtw_'+agg_app_norm[i][0]+'_norm.npy')

    ts_viz_conf['title'] = 'K-Medoids: DTW'

    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1],
        dist_matrix=dist_matrix,
        n_clusters=2,
        clust_func=KMedoids,
        metric='precomputed',
        method='pam',
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='dtw',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """


    """
    # KMedoids - DTW with Sakoe Chiba r=2
    print("\n** KMedoids - DTW with Sakoe Chiba r=2")

    dist_matrix = load(dist_matrix_path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')

    ts_viz_conf['title'] = 'K-Medoids: DTW w/ Sakoe Chiba r=2'

    values_list = []
    for j in range(5):
        elapsed_time = wms_distribution(
            df_norm_pkl,
            agg_app_norm[i][1],
            dist_matrix=dist_matrix,
            n_clusters=k,
            clust_func=KMedoids,
            metric='precomputed',
            method='pam',
            approach=agg_app_norm[i][0],
            path=path,
            clst_func_str='dtw_sakoe2_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )

        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)
    
    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
    """


    """
    # KMedoids - Soft-DTW
    print("** KMedoids - Soft-DTW")

    dist_matrix = load(dist_matrix_path+'dist_mat_soft_dtw_'+agg_app_norm[i][0]+'_norm.npy')

    ts_viz_conf['title'] = 'K-Medoids: Soft-DTW'

    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1],
        dist_matrix=dist_matrix,
        n_clusters=2,
        clust_func=KMedoids,
        metric='precomputed',
        method='pam',
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='soft_dtw',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """


    """
    #* Hierarchical *#
    linkage_method = 'Ward'

    ts_viz_conf['barycenter'] = False

    path = approach_path + 'Hierarchical/' + agg_app_norm[i][0] + '/'
    linkage_path = savehere = 'results/NORM/Approach_1/Hierarchical/'+ linkage_method + '/' \
        + agg_app_norm[i][0] + '/' \
        + agg_app_norm[i][0] + '_eucl'

    with open(linkage_path + '.pkl', 'rb') as f:
        linkage_matrix = pickle.load(f)

    # Hierarquical - Euclidean
    print("\n\n** Hierarquical - Euclidean")

    ts_viz_conf['title'] = 'Hierarchical: Euclidean - ' + linkage_method

    values_list = []
    for j in range(5):
        elapsed_time = wms_distribution(
            df_norm_pkl,
            agg_app_norm[i][1].squeeze(),
            n_clusters=k,
            clust_func=fcluster,
            linkage_matrix=linkage_matrix,
            criterion='maxclust',
            t=k,
            approach=agg_app_norm[i][0],
            path=path,
            clst_func_str='eucl_'+linkage_method+'_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )
        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)
    
    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
    """  

    """
    # Hierarquical - DTW
    print("** Hierarquical - DTW")

    linkage_path = savehere = 'results/NORM/Approach_1/Hierarchical/Ward/' \
        + agg_app_norm[i][0] + '/' \
        + agg_app_norm[i][0] + '_dtw'

    with open(linkage_path + '.pkl', 'rb') as f:
        linkage_matrix = pickle.load(f)


    ts_viz_conf['title'] = 'Hierarchical: DTW - Ward'
    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1].squeeze(),
        n_clusters=3,
        clust_func=fcluster,
        linkage_matrix=linkage_matrix,
        criterion='maxclust',
        t=3,
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='dtw_ward',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """

    """
    # Hierarquical - DTW w/ Sakoe Chiba r=2
    print("\n** Hierarquical - DTW w/ Sakoe Chiba r=2")

    linkage_path = savehere = 'results/NORM/Approach_1/Hierarchical/'+ linkage_method + '/' \
        + agg_app_norm[i][0] + '/' \
        + agg_app_norm[i][0] + '_dtw_sakoe2'

    with open(linkage_path + '.pkl', 'rb') as f:
        linkage_matrix = pickle.load(f)


    ts_viz_conf['title'] = 'Hierarchical: DTW w/ Sakoe Chiba r=2 - '+ linkage_method
    
    values_list = []
    for j in range(5):
        elapsed_time = wms_distribution(
            df_norm_pkl,
            agg_app_norm[i][1].squeeze(),
            n_clusters=k,
            clust_func=fcluster,
            linkage_matrix=linkage_matrix,
            criterion='maxclust',
            t=k,
            approach=agg_app_norm[i][0],
            path=path,
            clst_func_str='dtw_sakoe2_'+linkage_method+'_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )

        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)
    
    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
    """ 
    

    """
    # Hierarquical - Soft-DTW
    print("** Hierarquical - Soft-DTW")

    linkage_path = savehere = 'results/NORM/Approach_1/Hierarchical/Ward/' \
        + agg_app_norm[i][0] + '/' \
        + agg_app_norm[i][0] + '_soft_dtw'

    with open(linkage_path + '.pkl', 'rb') as f:
        linkage_matrix = pickle.load(f)


    ts_viz_conf['title'] = 'Hierarchical: Soft-DTW - Ward'
    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1].squeeze(),
        n_clusters=3,
        clust_func=fcluster,
        linkage_matrix=linkage_matrix,
        criterion='maxclust',
        t=3,
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='soft_dtw_ward',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """


    
    # Approach 2 #
    ##############
    print("## Approach 2 ##")
    approach_path = 'distribution/NORM/Approach_2/'

    ts_viz_conf['barycenter'] = False

    # KMeans - Euclidean
    print("** KMeans - Euclidean")
    path = approach_path + 'KMeans/' + agg_app_norm[i][0] + f'/k{k}/'

    ts_viz_conf['title'] = 'K-Means: Euclidean'

    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1],
        n_clusters=k,
        clust_func=KMeans,
        df_A2=agg_app_norm[i][2],
        intervals=[0,8,11,17,24],
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='A2',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )

    """
    # KMedoids - Euclidean
    print("** KMedoids - Euclidean")
    path = approach_path + 'KMedoids/' + agg_app_norm[i][0] + '/'

    ts_viz_conf['title'] = 'K-Medoids: Euclidean'

    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1].squeeze(),
        n_clusters=3,
        clust_func=KMedoids,
        metric='euclidean',
        method='pam',
        df_A2=agg_app_norm[i][2],
        intervals=[0,8,11,17,24],
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='A2',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )

  
    # Hierarchical - A2
    print("** Hierarchical - A2")

    ts_viz_conf['barycenter'] = False

    path = approach_path + 'Hierarchical/' + agg_app_norm[i][0] + '/'
    linkage_path = savehere = 'results/NORM/Approach_2/Hierarchical/Ward/' \
        + agg_app_norm[i][0] + '/' \
        + agg_app_norm[i][0] + '_eucl'

    with open(linkage_path + '.pkl', 'rb') as f:
        linkage_matrix = pickle.load(f)


    ts_viz_conf['title'] = 'Hierarchical: Euclidean - Ward'
    wms_distribution(
        df_norm_pkl,
        agg_app_norm[i][1].squeeze(),
        n_clusters=3,
        clust_func=fcluster,
        linkage_matrix=linkage_matrix,
        criterion='maxclust',
        t=3,
        approach=agg_app_norm[i][0],
        path=path,
        clst_func_str='euclidean_ward',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )
    """

#%%
agg_app_norm[i][2].info()


#%%

#                     RAW DATA                     #
####################################################

df_raw_pkl = pd.read_pickle("Data/dfnotnor.pkl")

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


ts_viz_conf = {
    'barycenter'    : True,
    'average'       : True,
    'dynamic_limits': True,
    'text'          : True,
    'save'          : True,
}

k = 2

for i in [1]:
    print(f"\n\n\n########   {agg_app_raw[i][0]}  ########")
    
    # Approach 2 #
    ##############
    print("## Approach 2 ##")
    approach_path = 'distribution/RAW/Approach_2/'

    ts_viz_conf['barycenter'] = False

    # KMeans - Euclidean
    print("** KMeans - Euclidean")
    path = approach_path + 'KMeans/' + agg_app_raw[i][0] + '/'

    ts_viz_conf['title'] = 'K-Means: Euclidean'

    wms_distribution(
        df_raw_pkl,
        agg_app_raw[i][1],
        n_clusters=k,
        clust_func=KMeans,
        df_A2=agg_app_raw[i][2],
        intervals=[0,8,11,17,24],
        approach=agg_app_raw[i][0],
        path=path,
        clst_func_str='A2',
        save_dist_txt=True,
        ts_viz=True,
        ts_viz_conf=ts_viz_conf,
    )


    # KMedoids - DTW with Sakoe Chiba r=2
    print("\n** KMedoids - DTW with Sakoe Chiba r=2")

    path = approach_path + 'KMedoids/' + agg_app_raw[i][0] + '/'

    dist_matrix_path = 'tests/dist_mat/'
    dist_matrix = load(dist_matrix_path+'dist_mat_dtw_sakoe2_'+agg_app_raw[i][0]+'_raw.npy')

    ts_viz_conf['title'] = 'K-Medoids: DTW w/ Sakoe Chiba r=2'

    wms_distribution(
            df_raw_pkl,
            agg_app_raw[i][1],
            dist_matrix=dist_matrix,
            n_clusters=k,
            clust_func=KMedoids,
            metric='precomputed',
            method='pam',
            approach=agg_app_raw[i][0],
            path=path,
            clst_func_str='dtw_sakoe2_k'+str(k),
            save_dist_txt=True,
            ts_viz=True,
            ts_viz_conf=ts_viz_conf,
        )


#%%
"""
#%%
# CV - Coefficient of Variation #
#################################
with open('Data/df_cv.pkl', 'rb') as f:
    df_cv = pickle.load(f)

with open("Data/dfnotnor.pkl", 'rb') as f:
    df_raw_pkl = pd.read_pickle(f)


print("## CV - Coefficent of variation ##")
approach_path = 'distribution/RAW/CV/'

ts_viz_conf['barycenter'] = False

#%%
# KMeans - Euclidean
print("** KMeans - Euclidean")
path = approach_path + 'KMeans/'

ts_viz_conf['title'] = 'K-Means: Euclidean | cv & avg'

wms_distribution(
    df_raw_pkl,
    agg_app_raw[i][1],
    n_clusters=3,
    clust_func=KMeans,
    df_A2=df_cv[['cv', 'avg']],
    approach='CV',
    path=path,
    clst_func_str='kmeans_cv__avg',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

#%%
# KMedoids - Euclidean
print("** KMedoids - Euclidean")
path = approach_path + 'KMedoids/'

ts_viz_conf['title'] = 'K-Medoids: Euclidean | cv & avg'

wms_distribution(
    df_raw_pkl,
    agg_app_raw[i][1],
    n_clusters=3,
    clust_func=KMedoids,
    metric='euclidean',
    method='pam',
    df_A2=df_cv[['cv', 'avg']],
    approach='CV',
    path=path,
    clst_func_str='kmedoids_cv__avg',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)


# Hierarchical
print("** Hierarchical")
path = approach_path + 'Hierarchical/'

"""





#%%
"""
ts_viz_conf = {
    'barycenter'    : True,
    'average'       : True,
    'dynamic_limits': True,
    'title'         : 'DTW w/ Sakoe Chiba r=2',
    'text'          : False,
    'save'          : True,
}


#%%
wms_distribution(
    df_norm_pkl,
    agg_app_norm[i][1],
    n_clusters=2,
    clust_func=TimeSeriesKMeans,
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    approach=agg_app_norm[i][0],
    path='distribution/'+agg_app_norm[i][0]+'/',
    clst_func_str='dtw_sakoe2',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

# %%
ts_viz_conf['title'] = 'DTW'

wms_distribution(
    df_norm_pkl,
    agg_app_norm[i][1],
    n_clusters=2,
    clust_func=TimeSeriesKMeans,
    metric="dtw",
    approach=agg_app_norm[i][0],
    path='distribution/'+agg_app_norm[i][0]+'/',
    clst_func_str='dtw',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

# %%
ts_viz_conf['title'] = 'Euclidean'

wms_distribution(
    df_norm_pkl,
    agg_app_norm[i][1],
    n_clusters=2,
    clust_func=TimeSeriesKMeans,
    metric="euclidean",
    approach=agg_app_norm[i][0],
    path='distribution/'+agg_app_norm[i][0]+'/',
    clst_func_str='euclidean',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)
# %%
ts_viz_conf['title'] = 'Soft-DTW'

wms_distribution(
    df_norm_pkl,
    agg_app_norm[i][1],
    n_clusters=2,
    clust_func=TimeSeriesKMeans,
    metric="softdtw",
    approach=agg_app_norm[i][0],
    path='distribution/'+agg_app_norm[i][0]+'/',
    clst_func_str='softdtw',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

#%%
ts_viz_conf['title']      = 'K-Means - Approach 2'
ts_viz_conf['barycenter'] = False

wms_distribution(
    df_norm_pkl,
    agg_app_norm[i][1],
    n_clusters=2,
    clust_func=KMeans,
    df_A2=agg_app_norm[i][2],
    intervals=[0,8,11,17,24],
    approach=agg_app_norm[i][0],
    path='distribution/'+agg_app_norm[i][0]+'/',
    clst_func_str='A2',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

#%%
# Coefficient of Variation #
############################

with open('Data/df_cv.pkl', 'rb') as f:
    df_cv = pickle.load(f)

with open("Data/dfnotnor.pkl", 'rb') as f:
    df_raw_pkl = pd.read_pickle(f)

# Load the time-series and dataframe objects:
with open('Data/ts_df.pkl', 'rb') as f:
    ts_gdc, df_gdc, ts_hdc, df_hdc, ts_hmdc, df_hmdc, ts_hwdc, df_hwdc = pickle.load(f)


#%%
ts_viz_conf['title']      = 'Coefficient of variation'
ts_viz_conf['barycenter'] = False

wms_distribution(
    df_raw_pkl,
    ts_hdc,
    n_clusters=8,
    clust_func=KMeans,
    df_A2=df_cv[['avg', 'cv']],
    approach='CV',
    path='distribution/CV/',
    clst_func_str='k8',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

# %%
df_cv.columns[0]
# %%
"""