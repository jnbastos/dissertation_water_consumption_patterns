#%%
from utils_clustering import *
import seaborn as sns; sns.set()  # for plot styling
import pickle
import numpy as np
from numpy import load



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


#                 RAW DATA                  #
#############################################

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

"""

#%%
i = 3

#%%
# Approach 1 #
##############
print("## Approach 1 ##")
approach_path = 'distribution/NORM/Approach_1/'

ts_viz_conf = {
    'barycenter'    : True,
    'average'       : True,
    'dynamic_limits': True,
    'text'          : False,
    'save'          : True,
}


df = df_norm_pkl
ts = agg_app_norm[i][1]
n_clusters = 3
random_state=RANDOM_SEED
clust_func=TimeSeriesKMeans
metric="dtw"
metric_params={'sakoe_chiba_radius': 2}

#%%
# Approach HWDC
df_aux = df.sort_values(by=['Local', 'Dia_da_semana'])
df_aux.drop_duplicates(subset=['Local', 'Dia_da_semana'], inplace=True)
df_aux = df_aux[['Dia','Dia_da_semana']]


#%%
# Time Series 
start = datetime.datetime.now()

km = clust_func(n_clusters=n_clusters, init='k-means++', random_state=random_state, metric_params=metric_params, metric=metric)
#km = clust_func(n_clusters=n_clusters, init=centroids, random_state=random_state, metric_params=metric_params, metric=metric)


km.fit(ts)
labels = km.predict(ts)

stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

#%%
centroids = km.cluster_centers_
centroids

#%%
for yi in range(n_clusters):
    plt.figure()
    centroids_ = centroids[yi].ravel()
    plt.plot(centroids_, "orange", linewidth=2, label='centróide')

    ts_cluster = ts[labels == yi]
    barycenter = np.zeros(24)
    barycenter = dtw_barycenter_averaging(ts_cluster, metric_params={'sakoe_chiba_radius': 2}, init_barycenter=centroids_)
    plt.plot(barycenter, "blue", linewidth=2, label='barycenter')

    plt.legend()
#%%
save1 = centroids

#%%
type(save1)
# %%
np.array_equal(save1, centroids)
# %%
bary1 = dtw_barycenter_averaging(ts_cluster, metric_params={'sakoe_chiba_radius': 2})
plt.plot(bary1, "blue", linewidth=2, label='barycenter 1')

bary2 = dtw_barycenter_averaging(ts_cluster, metric_params={'sakoe_chiba_radius': 2})
plt.plot(bary2, "yellow", linewidth=2, label='barycenter 2')

bary3 = dtw_barycenter_averaging(ts_cluster, metric_params={'sakoe_chiba_radius': 2}, init_barycenter=centroids_)
plt.plot(bary3, "red", linewidth=2, label='barycenter 3')

bary4 = dtw_barycenter_averaging(ts_cluster, metric_params={'sakoe_chiba_radius': 2})
plt.plot(bary4, "orange", linewidth=2, label='barycenter 4')

plt.legend()
# %%
np.array_equal(bary3,bary4)

"""
# %%
from tslearn.metrics import gamma_soft_dtw

n_clusters = 3
random_state=RANDOM_SEED

for i in [0,1,2,3,5]:
    ts = agg_app_norm[i][1]
    print(agg_app_norm[i][0])
    
    print(f"euclidean")
    start = datetime.datetime.now()
    euclidean_barycenter(ts)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}\n")

    print(f"dtw")
    start = datetime.datetime.now()
    dtw_barycenter_averaging(ts)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}\n")

    print(f"dtw w/sc r=2")
    start = datetime.datetime.now()
    dtw_barycenter_averaging(ts, metric_params={'sakoe_chiba_radius': 2})
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}\n")

    print(f"soft-dtw")
    start = datetime.datetime.now()
    gamma = gamma_soft_dtw(ts, n_samples=100, random_state=RANDOM_SEED)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"gamma time elapsed: {elapsed_time}")

    start = datetime.datetime.now()
    softdtw_barycenter(ts)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}\n\n")
    