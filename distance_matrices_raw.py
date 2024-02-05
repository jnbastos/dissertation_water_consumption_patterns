import sys
import datetime
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

#metrics = ['dtw', 'dtw_sakoe2', 'softdtw']
metrics = ['euclidean_A2', 'euclidean']


################################################################################


# Load the time-series and dataframe objects:
with open('Data/ts_df.pkl', 'rb') as f:
    ts_gdc_raw, df_gdc_raw, ts_hdc_raw, df_hdc_raw, ts_hmdc_raw, df_hmdc_raw, ts_hwdc_raw, df_hwdc_raw = pickle.load(f)

with open('Data/ts_df_2.pkl', 'rb') as f:
    ts_dc_raw, df_dc_raw, ts_hsdc_raw, df_hsdc_raw, ts_hwmdc_raw, df_hwmdc_raw = pickle.load(f)


agg_app_raw = [
    ('GDC', ts_gdc_raw, df_gdc_raw),       # 0
    ('HDC', ts_hdc_raw, df_hdc_raw),       # 1
    ('HMDC', ts_hmdc_raw, df_hmdc_raw),    # 2
    ('HWDC', ts_hwdc_raw, df_hwdc_raw),    # 3
    ('HWMDC', ts_hwmdc_raw, df_hwmdc_raw), # 4
    ('HSDC', ts_hsdc_raw, df_hsdc_raw),    # 5
    ('DC', ts_dc_raw, df_dc_raw)           # 6
    ]

path = 'tests/dist_mat/'

for i in range(len(agg_app_raw)):
    print(f"\n\n i = {i}")
    ts = agg_app_raw[i][1].reshape(-1, 24)

    # if neither HWMDC or DC
    if i not in [4,6]:
        if 'euclidean' in metrics:
            print(f"\nDistance Matrix Euclidean: {agg_app_raw[i][0]}")

            now = datetime.datetime.now()
            # Creates a condensed distance matrix
            distance_matrix_eucl = pdist(ts, 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_eucl_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_eucl)

        if 'dtw' in metrics:
            print(f"\nDistance Matrix DTW: {agg_app_raw[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw = cdist_dtw(
                ts, 
                n_jobs=-1, 
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_dtw)

        if 'dtw_sakoe2' in metrics:
            print(f"\nDistance Matrix DTW Sakoe r=2: {agg_app_raw[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw_sakoe2 = cdist_dtw(
                ts, 
                sakoe_chiba_radius=2,
                n_jobs=-1,
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_sakoe2_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_dtw_sakoe2)

        if 'softdtw' in metrics:
            # Run without paralelism (doesn't have the option)
            print(f"\nDistance Matrix Soft-DTW: {agg_app_raw[i][0]}")
            gamma = gamma_soft_dtw(ts, n_samples=100, random_state=RANDOM_SEED)
            print(f'Suggested gamma: {gamma}')

            now = datetime.datetime.now()
            #distance_matrix_soft_dtw = cdist_soft_dtw(ts)
            distance_matrix_soft_dtw = cdist_soft_dtw_normalized(ts, gamma=gamma)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_soft_dtw_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_soft_dtw)

        # Approach 2 distance matrices
        if 'euclidean_A2' in metrics:
            print(f"\nDistance Matrix Euclidean - Approach 2: {agg_app_raw[i][0]}")
            now = datetime.datetime.now()
            # Creates a condensed distance matrix
            distance_matrix_eucl_A2 = pdist(agg_app_raw[i][2], 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_euclA2_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_eucl_A2)


    # if HWMDC or DC
    else:
        if ('euclidean' in metrics) or ('dtw' in metrics) or ('dtw_sakoe2' in metrics) or ('softdtw' in metrics):
            '''
            # Permutation of the matrix and ts dataset downsized to 10000 days
            ts_perm = np.random.permutation(ts)[0:10000]

            with open('Data/ts_perm_'+agg_app_raw[i][0]+'raw.pkl', 'wb') as f:
                pickle.dump(ts_perm, f)
            '''
            with open('Data/ts_perm_'+agg_app_raw[i][0]+'raw.pkl', 'rb') as f:
                ts_perm = pickle.load(f)

        if 'euclidean' in metrics:
            print(f"\nDistance Matrix Euclidean: {agg_app_raw[i][0]}")

            now = datetime.datetime.now()
            # Creates a condensed distance matrix
            distance_matrix_eucl = pdist(ts_perm.reshape(-1,24), 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_eucl_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_eucl)

        if 'dtw' in metrics:
            print(f"\nDistance Matrix DTW: {agg_app_raw[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw = cdist_dtw(
                ts_perm, 
                n_jobs=-1, 
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_dtw)

        if 'dtw_sakoe2' in metrics:
            print(f"\nDistance Matrix DTW Sakoe r=2: {agg_app_raw[i][0]}")

            now = datetime.datetime.now()
            distance_matrix_dtw_sakoe2 = cdist_dtw(
                ts_perm, 
                sakoe_chiba_radius=2,
                n_jobs=-1,
                verbose=0)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_dtw_sakoe2_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_dtw_sakoe2)

        if 'softdtw' in metrics:
            print(f"\nDistance Matrix Soft-DTW: {agg_app_raw[i][0]}")
            gamma = gamma_soft_dtw(ts_perm, n_samples=100, random_state=RANDOM_SEED)
            print(f'Suggested gamma: {gamma}')

            now = datetime.datetime.now()
            #distance_matrix_soft_dtw = cdist_soft_dtw(ts_perm)
            distance_matrix_soft_dtw = cdist_soft_dtw_normalized(ts_perm, gamma=gamma)
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_soft_dtw_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_soft_dtw)

        # Approach 2 distance matrices and permutated DataFrames creation
        if 'euclidean_A2' in metrics:
            '''
            df_perm = agg_app_raw[i][2].sample(n=10000, random_state=RANDOM_SEED)
            with open('Data/df_perm_'+agg_app_raw[i][0]+'_raw.pkl', 'wb') as f:
                pickle.dump(df_perm, f)
            '''
            with open('Data/df_perm_'+agg_app_raw[i][0]+'_raw.pkl', 'rb') as f:
                df_perm = pickle.load(f)

            print(f"\nDistance Matrix Euclidean - Approach 2: {agg_app_raw[i][0]}")
            now = datetime.datetime.now()

            # Creates a condensed distance matrix
            distance_matrix_eucl_A2 = pdist(df_perm, 'euclidean')
            time_stop = datetime.datetime.now()
            print(f"time elapsed: {time_stop - now}")

            save(path+'dist_mat_euclA2_'+agg_app_raw[i][0]+'_raw.npy', distance_matrix_eucl_A2)
        


