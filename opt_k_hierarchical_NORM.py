#%%
#import pandas as pd
import seaborn as sns
import pickle
import datetime

from utils_optimal_k import *
from numpy import load
from scipy.spatial.distance import squareform

import sys

# Temporary so it not raise de error:
# RecursionError: maximum recursion depth exceeded while getting the str of an object
sys.setrecursionlimit(10000)

sns.set_style("darkgrid")

RANDOM_SEED = 42


#%%
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
dist_path = 'tests/dist_mat/'

#%%
print("########### HIERARCHICAL ###########")

"""
for i in range(7):
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    # Dendograms
    for method in ['complete', 'single', 'average', 'ward', 'median', 'centroid', 'weighted']:
        print(f"\n\n ### {method} ###")
        for metric in ['dtw', 'dtw_sakoe2', 'soft_dtw']:
            print(f"\n** {metric.upper()} **")
            distance_matrix = load(dist_path + 'dist_mat_' + metric + '_' + agg_app_norm[i][0] + '_norm.npy')
            savehere = 'results/NORM/Approach_1/Hierarchical/' + method.capitalize() + '/' + agg_app_norm[i][0] + '/' + agg_app_norm[i][0] + '_' + metric
            
            start = datetime.datetime.now()
            linkage_matrix = hierarchical_clustering(squareform(distance_matrix), method=method, save=savehere)
            stop = datetime.datetime.now()
            elapsed_time = str(stop - start)
            print(f"time elapsed: {elapsed_time}")

            with open(savehere + '.pkl', 'wb') as f:
                pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)
"""

print("### HIERACHICAL (NORM) EUCLIDEAN - Approach 1 and 2")
#for i in [range(7)]:
for i in [0,1,2,3,5]:
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")
    # Dendograms

    #for method in ['complete', 'single', 'average', 'ward', 'median', 'centroid', 'weighted']:
    for method in ['ward']:
        print(f"\n\n ### {method} ###")
        """
        # APROACH 1 #
        print("Approach 1\n")

        metric = 'eucl'
        print(f"\n** {metric.upper()} **")

        print("### Approach 1 - Euclidean ###")
        distance_matrix = load(dist_path + 'dist_mat_' + metric + '_' + agg_app_norm[i][0] + '_norm.npy')
        savehere = 'results/NORM/Approach_1/Hierarchical/' + method.capitalize() + '/' + agg_app_norm[i][0] + '/' + agg_app_norm[i][0] + '_' + metric
        
        start = datetime.datetime.now()
        linkage_matrix = hierarchical_clustering(distance_matrix, method=method, save=savehere)
        stop = datetime.datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        with open(savehere + '.pkl', 'wb') as f:
            pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)
        """

        # APROACH 2 #
        print("Approach 2")

        metric = 'eucl'
        print(f"\n** {metric.upper()} **")

        print("### Approach 2 - Euclidean ###")
        distance_matrix = load(dist_path + 'dist_mat_' + metric + '_' + agg_app_norm[i][0] + '_norm.npy')
        savehere = 'results/NORM/Approach_2/Hierarchical/' + method.capitalize() + '/' + agg_app_norm[i][0] + '/' + agg_app_norm[i][0] + '_' + metric
        
        start = datetime.datetime.now()
        linkage_matrix = hierarchical_clustering(distance_matrix, method=method, save=savehere)
        stop = datetime.datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        #with open(savehere + '.pkl', 'wb') as f:
        #    pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)


# %%
