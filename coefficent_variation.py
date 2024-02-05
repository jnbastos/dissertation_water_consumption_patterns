#%%
import pickle
import datetime

from utils_clustering import *
from utils_optimal_k import *
from numpy import load
from numpy import save
from scipy.spatial.distance import pdist




#%%
# Load raw data
df_pkl = pd.read_pickle("Data/dfnotnor.pkl")

# Get DataFrame with information from each one of the households
# Columns: Local, daily_sd, sd, avg, cv
df_cv = cv(df_pkl)

# Save DataFrame
df_cv.to_pickle("Data/df_cv.pkl")

#%%
df_cv
# %%

clusters_k = range(2,51)

print("\n\n######### COEFFICIENT OF VARIATION #########")

dist_path = 'tests/dist_mat/'

columns = [['cv', 'avg'], ['sd', 'avg'], ['daily_sd', 'avg']]
str_col = ["cv__avg", "sd__avg", "daily_sd__avg"]

for i in range(len(columns)):
    print(f"\n\n########   {str_col[i]}  ########")


    # KMEANS
    print(f"\nKMEANS - {str_col[i]}")
    savehere = 'results/RAW/CV/KMeans/'+str_col[i]

    print("\n SILHOUETTE\n")
    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        df_cv[columns[i]],
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

        

    print("\nCALINSKY HARABASZ\n")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
            df_cv[columns[i]], 
            clusters_k,
            KMeans,
            metric="euclidean",
            save=savehere+'_harabasz',
            metric_yb='calinski_harabasz',
            locate_elbow=False
            )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_harabasz.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))


    """
    print("\n# ELBOW")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
        df_cv[columns[i]],
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
        

    # KMEDOIDS
    print(f"\nKMEDOIDS - {str_col[i]}")
    savehere = 'results/RAW/CV/KMedoids/'+str_col[i]

    print("\n# SILHOUETTE")
    start = datetime.datetime.now()
    avg_silhouettes, inertia, n_iter = silhouette(
        df_cv[columns[i]],
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

        
    print("\nCALINSKY HARABASZ\n")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
            df_cv[columns[i]], 
            clusters_k,
            KMedoids,
            metric="euclidean",
            save=savehere+'_harabasz',
            metric_yb='calinski_harabasz',
            locate_elbow=False
            )
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    with open(savehere+'_harabasz.npy', 'wb') as f:
        np.save(f, np.array([elapsed_time, k_scores, k_timers, elbow_value, elbow_score], dtype=object))
        

    """
    print("\n# ELBOW")
    start = datetime.datetime.now()
    k_scores, k_timers, elbow_value, elbow_score = elbow(
        df_cv[columns[i]],
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

    
    # HIERARCHICAL
    print("#### HIERARQUICAL ####")
    print(f"\nDistance Matrix Euclidean - CV (columns: {str_col[i]})")
    now = datetime.datetime.now()
    # Creates a condensed distance matrix
    distance_matrix_cv = pdist(df_cv[columns[i]], 'euclidean')
    time_stop = datetime.datetime.now()
    print(f"time elapsed: {time_stop - now}")

    save(dist_path+'dist_mat_CV_'+str_col[i]+'.npy', distance_matrix_cv)

    print("### Coefficient of Variation - Hierarquical ###")
    for method in ['complete', 'single', 'average', 'ward', 'median', 'centroid', 'weighted']:
        print(f"\n\n ### {method} ###")

        metric = 'eucl'
        print(f"\n** {metric.upper()} **")

        distance_matrix_cv = load(dist_path+'dist_mat_CV_'+str_col[i]+'.npy')
        savehere = 'results/RAW/CV/Hierarchical/' + method.capitalize() + '/' + str_col[i] + '_' + metric
        
        start = datetime.datetime.now()
        linkage_matrix = hierarchical_clustering(distance_matrix_cv, method=method, save=savehere)
        stop = datetime.datetime.now()
        elapsed_time = str(stop - start)
        print(f"time elapsed: {elapsed_time}")

        with open(savehere + '.pkl', 'wb') as f:
            pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)

    """


"""
    #%%
    # [['sd', 'avg']]
    #Best score: 0.48438539726124674 with k = 2

    # [['daily_sd', 'avg']]
    # Best score: 0.58347945232647 with k = 4

    # [['cv', 'avg']]
    # Best score: 0.48438539726124674 with k = 2
    silhouette(
        df_cv[['cv', 'avg']], 
        clusters_k,
        KMeans,
        sil_metric="euclidean",
        save_txt='./results/CV_kmeans_cv_avg',
        save_all_results=True,
        )
    # %%
    # ['sd', 'avg']  =>  k = 9
    # ['daily_sd', 'avg'] => k = 8
    # ['cv', 'avg'] => k = 8
    elbow(
        df_cv[['cv', 'avg']],
        clusters_k,
        KMeans,
        metric="euclidean",
        save='./results/CV_elbow_kmeans_cv_avg'
        )
"""

"""
#%%
# Open results [.npy]
openhere = 'results/RAW/CV/KMedoids/daily_sd__avg_harabasz.npy'

with open(openhere, 'rb') as f:
    # elapsed_time, k_scores, 
    # k_timers, elbow_value, elbow_score
    harabasz = np.load(f, allow_pickle=True)

# %%
harabasz.shape
# %%
np.max(harabasz[1])

#%%
np.argmax(harabasz[1])+2
# %%
len(harabasz[1])
# %%

# %%
"""


#%%
print(f"\nDistance Matrix Euclidean - Approach 3:")

now = datetime.datetime.now()
# Creates a condensed distance matrix
distance_matrix_eucl_A3_cv__avg = pdist(df_cv[['cv', 'avg']], 'euclidean')
time_stop = datetime.datetime.now()
print(f"time elapsed: {time_stop - now}")

path = 'tests/dist_mat/'
save(path+'dist_mat_euclA3_cv__avg.npy', distance_matrix_eucl_A3_cv__avg)


# %%
# Hierarquical Ward

print("### HIERACHICAL (RAW) EUCLIDEAN - Approach 3")
print(f"\n\n########   WARD LINKAGE  ########")
# Dendograms
method = 'ward'

print(f"\n\n ### {method} ###")

# APROACH 3 #
metric = 'eucl'
print(f"\n** {metric.upper()} **")

print("### Approach 3 - Euclidean ###")
distance_matrix = load(path+'dist_mat_euclA3_cv__avg.npy')
savehere = 'results/RAW/CV/Hierarchical/' + method.capitalize() + '/cv__avg2_' + metric

start = datetime.datetime.now()
linkage_matrix = hierarchical_clustering(distance_matrix, method=method, save=savehere)
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

with open(savehere + '.pkl', 'wb') as f:
    pickle.dump(linkage_matrix, f, pickle.HIGHEST_PROTOCOL)

# %%
clusters_k = [3,6,9,15,17,19,22]

print(f"\n## SILHOUETTE METHOD EUCLIDEAN w/ KMeans:")
start = datetime.datetime.now()
avg_silhouettes, inertia, n_iter = silhouette(
    df_cv[['cv', 'avg']],
    clusters_k,
    KMeans,
    init='k-means++',
    sil_metric="euclidean",
    imgs=True,
    save_img='results/RAW/CV/KMeans/cv__avg_'
)
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

# %%
