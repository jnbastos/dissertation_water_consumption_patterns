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
dist_path = 'tests/dist_mat/'

#%%
print("########### HIERARCHICAL - RAW ###########")


for i in range(7):
    print(f"\n\n########   {agg_app_raw[i][0]}  ########")
    # Dendograms
    for method in ['complete', 'single', 'average', 'ward', 'median', 'centroid', 'weighted']:
        print(f"\n\n ### {method} ###")
        for metric in ['dtw', 'dtw_sakoe2', 'soft_dtw']:
            print(f"\n** {metric.upper()} **")
            distance_matrix = load(dist_path + 'dist_mat_' + metric + '_' + agg_app_raw[i][0] + '_raw.npy')
            savehere = 'results/RAW/Hierarchical/' + method.capitalize() + '/' + agg_app_raw[i][0] + '/' + agg_app_raw[i][0] + '_' + metric
            
            start = datetime.datetime.now()
            linkage_matrix = hierarchical_clustering(squareform(distance_matrix), method=method, save=savehere)
            stop = datetime.datetime.now()
            elapsed_time = str(stop - start)
            print(f"time elapsed: {elapsed_time}")

            with open(savehere + '.pkl', 'wb') as f:
                pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)
            
"""


print("### HIERACHICAL (RAW) EUCLIDEAN - Approach 1 and 2")
for i in range(7):
    print(f"\n\n########   {agg_app_raw[i][0]}  ########")
    # Dendograms

    
    for method in ['complete', 'single', 'average', 'ward', 'median', 'centroid', 'weighted']:
        print(f"\n\n ### {method} ###")

        # APROACH 1 #
        metric = 'eucl'
        print(f"\n** {metric.upper()} **")

        print("### Approach 1 - Euclidean ###")
        distance_matrix = load(dist_path + 'dist_mat_' + metric + '_' + agg_app_raw[i][0] + '_raw.npy')
        savehere = 'results/RAW/Hierarchical/' + method.capitalize() + '/' + agg_app_raw[i][0] + '/' + agg_app_raw[i][0] + '_' + metric
        
        start = datetime.datetime.now()
        linkage_matrix = hierarchical_clustering(distance_matrix, method=method, save=savehere)
        stop = datetime.datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        with open(savehere + '.pkl', 'wb') as f:
            pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)

        # APROACH 2 #
        metric = 'eucl'
        print(f"\n** {metric.upper()} **")

        print("### Approach 2 - Euclidean ###")
        distance_matrix = load(dist_path + 'dist_mat_' + metric + '_' + agg_app_raw[i][0] + '_raw.npy')
        savehere = 'results/RAW/Approach_2/Hierarchical/' + method.capitalize() + '/' + agg_app_raw[i][0] + '/' + agg_app_raw[i][0] + '_' + metric
        
        start = datetime.datetime.now()
        linkage_matrix = hierarchical_clustering(distance_matrix, method=method, save=savehere)
        stop = datetime.datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        with open(savehere + '.pkl', 'wb') as f:
            pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)
"""