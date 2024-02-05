#%%
import sys
#import datetime

from datetime import datetime
from datetime import timedelta
from datetime import time


import numpy as np
import pickle
import pandas

from numpy import save
from tslearn.metrics import cdist_dtw
from tslearn.metrics import cdist_soft_dtw, cdist_soft_dtw_normalized, gamma_soft_dtw
from scipy.spatial.distance import pdist

RANDOM_SEED = 42

################################################################################
# PARAMS:

metrics = ['euclidean', 'dtw', 'dtw_sakoe2', 'softdtw']
#metrics = ['euclidean_A2']


################################################################################

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

path = 'tests/dist_mat/'

#%%
################################################################################
# Gamma running times #


for i in [0,1,2,3,5]:
    print(f"\n\n i = {i}")
    ts = agg_app_norm[i][1].reshape(-1, 24)
    values_list = []
    print(f"\nDistance Matrix Soft-DTW: {agg_app_norm[i][0]}")
    for j in range(5):
        now = datetime.now()
        gamma = gamma_soft_dtw(ts, n_samples=100, random_state=RANDOM_SEED)
        time_stop = datetime.now()
        
        if j==0:
            print(f'Suggested gamma: {gamma}')

        elapsed_time = time_stop - now
        print(f"time elapsed: {elapsed_time}")
        
        elapsed_time_ms = elapsed_time.total_seconds()*1000
        print(f"time elapsed (ms): {elapsed_time_ms}")

        values_list.append(elapsed_time_ms)

    arr = np.array(values_list)
    values_mean = np.mean(arr, axis=0)
    values_std = np.std(arr, axis=0)

    print(f'{agg_app_norm[i][0]}:\nmean = {values_mean}\nstd = {values_std}\n')
            

################################################################################
#%%


#%%
#for i in range(len(agg_app_norm)):
for i in [0,1,2,3,5]:    
    print(f"\n\n i = {i}")
    ts = agg_app_norm[i][1].reshape(-1, 24)

    # if neither HWMDC or DC
    if i not in [4,6]:
        if 'euclidean' in metrics:
            print(f"\nDistance Matrix Euclidean: {agg_app_norm[i][0]}")

            now = datetime.datetime.now()
            # Creates a condensed distance matrix
            distance_matrix_eucl = pdist(ts, 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_eucl_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_eucl)

        if 'dtw' in metrics:
            print(f"\nDistance Matrix DTW: {agg_app_norm[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw = cdist_dtw(
                ts, 
                n_jobs=1, 
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_dtw)

        if 'dtw_sakoe2' in metrics:
            print(f"\nDistance Matrix DTW Sakoe r=2: {agg_app_norm[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw_sakoe2 = cdist_dtw(
                ts, 
                sakoe_chiba_radius=2,
                n_jobs=1,
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_dtw_sakoe2)

        if 'softdtw' in metrics:
            # Run without paralelism (doesn't have the option)
            print(f"\nDistance Matrix Soft-DTW: {agg_app_norm[i][0]}")
            gamma = gamma_soft_dtw(ts, n_samples=100, random_state=RANDOM_SEED)
            print(f'Suggested gamma: {gamma}')

            now = datetime.datetime.now()
            #distance_matrix_soft_dtw = cdist_soft_dtw(ts)
            # https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.cdist_soft_dtw_normalized.html#tslearn.metrics.cdist_soft_dtw_normalized
            distance_matrix_soft_dtw = cdist_soft_dtw_normalized(ts, gamma=gamma)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_soft_dtw_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_soft_dtw)

        # Approach 2 distance matrices and permutated DataFrames creation
        if 'euclidean_A2' in metrics:
            print(f"\nDistance Matrix Euclidean - Approach 2: {agg_app_norm[i][0]}")
            now = datetime.datetime.now()
            # Creates a condensed distance matrix
            distance_matrix_eucl_A2 = pdist(agg_app_norm[i][2], 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_euclA2_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_eucl_A2)

    # if HWMDC or DC
    else:
        # TimeSeries (Approach 1) permutation creation
        if ('euclidean' in metrics) or ('dtw' in metrics) or ('dtw_sakoe2' in metrics) or ('softdtw' in metrics):
            '''
            # Permutation of the matrix and ts dataset downsized to 10000 days
            ts_perm = np.random.permutation(ts)[0:10000]

            with open('Data/ts_perm_'+agg_app_norm[i][0]+'_norm.pkl', 'wb') as f:
                pickle.dump(ts_perm, f)
            '''

            with open('Data/ts_perm_'+agg_app_norm[i][0]+'_norm.pkl', 'rb') as f:
                ts_perm = pickle.load(f)
            

        if 'euclidean' in metrics:
            print(f"\nDistance Matrix Euclidean: {agg_app_norm[i][0]}")

            now = datetime.datetime.now()
            # Creates a condensed distance matrix
            distance_matrix_eucl = pdist(ts_perm.reshape(-1,24), 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_eucl_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_eucl)

        if 'dtw' in metrics:
            print(f"\nDistance Matrix DTW: {agg_app_norm[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw = cdist_dtw(
                ts_perm, 
                n_jobs=-1, 
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_dtw)

        if 'dtw_sakoe2' in metrics:
            print(f"\nDistance Matrix DTW Sakoe r=2: {agg_app_norm[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw_sakoe2 = cdist_dtw(
                ts_perm, 
                sakoe_chiba_radius=2,
                n_jobs=-1,
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_dtw_sakoe2)

        if 'softdtw' in metrics:
            print(f"\nDistance Matrix Soft-DTW: {agg_app_norm[i][0]}")
            gamma = gamma_soft_dtw(ts_perm, n_samples=100, random_state=RANDOM_SEED)
            print(f'Suggested gamma: {gamma}')
            
            now = datetime.datetime.now()
            #distance_matrix_soft_dtw = cdist_soft_dtw(ts_perm)
            distance_matrix_soft_dtw = cdist_soft_dtw_normalized(ts_perm, gamma=gamma)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_soft_dtw_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_soft_dtw)

        # Approach 2 distance matrices and permutated DataFrames creation
        if 'euclidean_A2' in metrics:
            '''
            df_perm = agg_app_norm[i][2].sample(n=10000, random_state=RANDOM_SEED)
            with open('Data/df_perm_'+agg_app_norm[i][0]+'_norm.pkl', 'wb') as f:
                pickle.dump(df_perm, f)
            '''

            with open('Data/df_perm_'+agg_app_norm[i][0]+'_norm.pkl', 'rb') as f:
                df_perm = pickle.load(f)

            print(f"\nDistance Matrix Euclidean - Approach 2: {agg_app_norm[i][0]}")
            now = datetime.datetime.now()

            # Creates a condensed distance matrix
            distance_matrix_eucl_A2 = pdist(df_perm, 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_euclA2_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_eucl_A2)
        
#%%
# A2 distance matrix for dtw_sakoe2

for i in [0,1,2,3,5]:
    print(f"\n\n i = {i}")
    print(f"\nDistance Matrix DTW Sakoe r=2: {agg_app_norm[i][0]}")

    #ts = agg_app_norm[i][2].reshape(-1, 5)

    now = datetime.now()
    distance_matrix_dtw_sakoe2 = cdist_dtw(
        agg_app_norm[i][2], 
        sakoe_chiba_radius=2,
        n_jobs=1,
        verbose=0)
    time_stop = datetime.now()
    print(f"time elapsed: {time_stop - now}")

    save(path+'dist_mat_A2_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy', distance_matrix_dtw_sakoe2)

# %%
