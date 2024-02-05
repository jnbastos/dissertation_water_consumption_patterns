#%%
# Conda env: gap-stat
# ValueError: unsupported pickle protocol: 5
# Python 3.7.12 -> 3.8.12


import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from gap_statistic import OptimalK
import warnings

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Union, Iterable, Callable, Generator
from scipy.cluster.vq import kmeans2

import datetime
from numpy import load


RANDOM_SEED = 42


def tsk_euclidean(ts, k):
    
    km = TimeSeriesKMeans(
        metric="euclidean", 
        random_state=RANDOM_SEED,
        n_jobs=-1)
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)


def tsk_dtw(ts, k):

    km = TimeSeriesKMeans(
        metric="dtw", 
        random_state=RANDOM_SEED,
        n_jobs=-1)
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)


def tsk_dtw_sakoe_r2(ts, k):
    """ 
    Special clustering function which uses the TimeSeriesKMeans
    model from tslearn.
    
    These user defined functions *must* take the ts and a k 
    and can take an arbitrary number of other kwargs, which can
    be pass with `clusterer_kwargs` when initializing OptimalK
    """
    
    km = TimeSeriesKMeans(
        metric="dtw",
        metric_params={'sakoe_chiba_radius': 2}, 
        random_state=RANDOM_SEED,
        n_jobs=-1)
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)


def tsk_softdtw(ts, k):
    
    km = TimeSeriesKMeans(
        metric="softdtw", 
        random_state=RANDOM_SEED,
        n_jobs=-1)
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)


def kmeans_euclidean(ts, k):
    
    km = KMeans(
        random_state=RANDOM_SEED,
        n_init='auto',
        init='k-means++'
        )
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)


def kmedoids_euclidean(ts, k):
    
    km = KMedoids(
        random_state=RANDOM_SEED,
        method='pam',
        init = 'k-medoids++',
        metric='euclidean'
        )
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)


def kmedoids_precomputed(ts, k):
    km = KMedoids(
        random_state=RANDOM_SEED,
        method='pam',
        init = 'k-medoids++',
        metric='precomputed'
        )
    km.fit(ts)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return km.cluster_centers_, km.predict(ts)



try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_FOUND = True
except ImportError:
    MATPLOTLIB_FOUND = False
    warnings.warn("matplotlib not installed; results plotting is disabled.")


class OptimalK_v2(OptimalK):
    """
    Same as OptimalK, but with a few extra features:
    - The Gap Values by Cluster Count plot is now saved as a .png file.
    """
    def __call__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_refs: int = 3,
        cluster_array: Iterable[int] = (),
        save: str = None,
    ):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
        :param X - pandas dataframe or numpy array of data points of shape (n_samples, n_features)
        :param n_refs - int: Number of random reference data sets used as inertia reference to actual data.
        :param cluster_array - 1d iterable of integers; each representing n_clusters to try on the data.
        """

        # Convert the 1d array of n_clusters to try into an array
        # Raise error if values are less than 1 or larger than the unique sample in the set.
        cluster_array = np.array([x for x in cluster_array]).astype(int)
        if np.where(cluster_array < 1)[0].shape[0]:
            raise ValueError(
                "cluster_array contains values less than 1: {}".format(
                    cluster_array[np.where(cluster_array < 1)[0]]
                )
            )
        if cluster_array.shape[0] > X.shape[0]:
            raise ValueError(
                "The number of suggested clusters to try ({}) is larger than samples in dataset. ({})".format(
                    cluster_array.shape[0], X.shape[0]
                )
            )
        if not cluster_array.shape[0]:
            raise ValueError("The supplied cluster_array has no values.")

        # Array of resulting gaps.
        gap_df = pd.DataFrame({"n_clusters": [], "gap_value": []})

        # Define the compute engine; all methods take identical args and are generators.
        if self.parallel_backend == "joblib":
            engine = self._process_with_joblib
        elif self.parallel_backend == "multiprocessing":
            engine = self._process_with_multiprocessing
        elif self.parallel_backend == "rust":
            engine = self._process_with_rust
        else:
            engine = self._process_non_parallel

        # Calculate the gaps for each cluster count.
        for gap_calc_result in engine(X, n_refs, cluster_array):
            # Assign this loop's gap statistic to gaps
            gap_df = pd.concat(
                [
                    gap_df,
                    pd.DataFrame(
                        {
                            "n_clusters": [gap_calc_result.n_clusters],
                            "gap_value": [gap_calc_result.gap_value],
                            "ref_dispersion_std": [gap_calc_result.ref_dispersion_std],
                            "sdk": [gap_calc_result.sdk],
                            "sk": [gap_calc_result.sk],
                            "gap*": [gap_calc_result.gap_star],
                            "sk*": [gap_calc_result.sk_star],
                        }
                    ),
                ]
            )
            gap_df["gap_k+1"] = gap_df["gap_value"].shift(-1)
            gap_df["gap*_k+1"] = gap_df["gap*"].shift(-1)
            gap_df["sk+1"] = gap_df["sk"].shift(-1)
            gap_df["sk*+1"] = gap_df["sk*"].shift(-1)
            gap_df["diff"] = gap_df["gap_value"] - gap_df["gap_k+1"] + gap_df["sk+1"]
            gap_df["diff*"] = gap_df["gap*"] - gap_df["gap*_k+1"] + gap_df["sk*+1"]

        if save is not None:
            with open(save+'.pkl', 'wb') as f:
                pickle.dump([gap_df], f)

        # drop auxilariy columns
        gap_df.drop(
            labels=["sdk", "gap_k+1", "gap*_k+1", "sk+1", "sk*+1"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        self.gap_df = gap_df.sort_values(by="n_clusters", ascending=True).reset_index(
            drop=True
        )
        self.n_clusters = int(
            self.gap_df.loc[np.argmax(self.gap_df.gap_value.values)].n_clusters
        )
        return self.n_clusters


    def plot_results(self, save_img=False, img_path=""):
        """
        Plots the results of the last run optimal K search procedure.
        Four plots are printed:
        (1) A plot of the Gap value - as defined in the original Tibshirani et
        al paper - versus n, the number of clusters.
        (2) A plot of diff versus n, the number of clusters, where diff =
        Gap(k) - Gap(k+1) + s_{k+1}. The original Tibshirani et al paper
        recommends choosing the smallest k such that this measure is positive.
        (3) A plot of the Gap* value - a variant of the Gap statistic suggested
        in "A comparison of Gap statistic definitions with and with-out
        logarithm function" [https://core.ac.uk/download/pdf/12172514.pdf],
        which simply removes the logarithm operation from the Gap calculation -
        versus n, the number of clusters.
        (4) A plot of the diff* value versus n, the number of clusters. diff*
        corresponds to the aforementioned diff value for the case of Gap*.
        """
        if not MATPLOTLIB_FOUND:
            print("matplotlib not installed; results plotting is disabled.")
            return
        if not hasattr(self, "gap_df") or self.gap_df is None:
            print("No results to print. OptimalK not called yet.")
            return

        # Gap values plot
        plt.plot(self.gap_df.n_clusters, self.gap_df.gap_value, linewidth=3)
        plt.scatter(
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].n_clusters,
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].gap_value,
            s=250,
            c="r",
        )
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Gap Value")
        plt.title("Gap Values by Cluster Count")
        if save_img:
            plt.savefig(img_path + ".png")
            plt.clf()
        else:
            plt.show()

        # diff plot
        plt.plot(self.gap_df.n_clusters, self.gap_df["diff"], linewidth=3)
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Diff Value")
        plt.title("Diff Values by Cluster Count")
        plt.show()

        # Gap* plot
        max_ix = self.gap_df[self.gap_df["gap*"] == self.gap_df["gap*"].max()].index[0]
        plt.plot(self.gap_df.n_clusters, self.gap_df["gap*"], linewidth=3)
        plt.scatter(
            self.gap_df.loc[max_ix]["n_clusters"],
            self.gap_df.loc[max_ix]["gap*"],
            s=250,
            c="r",
        )
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Gap* Value")
        plt.title("Gap* Values by Cluster Count")
        plt.show()

        # diff* plot
        plt.plot(self.gap_df.n_clusters, self.gap_df["diff*"], linewidth=3)
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Diff* Value")
        plt.title("Diff* Values by Cluster Count")
        plt.show()

#%%
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
i = 3

n_refs = 100

for i in [0,1,2,3,5]:

    # removes dimensions of size 1
    ap_1_ts = agg_app_norm[i][1]
    ap_1 = ap_1_ts.squeeze()

    """
    # TIME SERIES KMEANS #
    print("######## TIME SERIES KMEANS - GAP STATISTIC ########")
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")

    
    # TSKMeans EUCLIDEAN
    ##########################
    print("\n\n TSKMeans EUCLIDEAN\n")

    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl_gap'

    start = datetime.datetime.now()
    optimalk = OptimalK_v2(clusterer=tsk_euclidean)
    n_clusters = optimalk(
        agg_app_norm[i][1], 
        n_refs=n_refs,
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")


    
    # TSKMeans DTW
    ##########################
    print("\n\n TSKMeans DTW\n")

    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_dtw_gap'

    optimalk = OptimalK_v2(clusterer=tsk_dtw)

    start = datetime.datetime.now()
    n_clusters = optimalk(
        agg_app_norm[i][1], 
        n_refs=n_refs, 
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")


    # TSKMeans DTW w/ Sakoe Chiba r=2
    ##########################
    print("\n\n TSKMeans DTW with Sakoe Chiba r=2\n")

    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_dtw_sakoe_2_gap'

    start = datetime.datetime.now()
    optimalk = OptimalK_v2(clusterer=tsk_dtw_sakoe_r2)
    n_clusters = optimalk(
        agg_app_norm[i][1], 
        n_refs=n_refs, 
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")


    # TSKMeans Soft-DTW
    ##########################
    print("\n\n TSKMeans Soft-DTW\n")

    savehere = 'results/NORM/Approach_1/TSKMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_softdtw_gap'

    optimalk = OptimalK_v2(clusterer=tsk_softdtw)

    start = datetime.datetime.now()
    n_clusters = optimalk(
        agg_app_norm[i][1], 
        n_refs=n_refs, 
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")
    """


    # TIME SERIES KMEDOIDS #
    print("######## TIME SERIES KMEDOIDS - GAP STATISTIC ########")
    print(f"\n\n########   {agg_app_norm[i][0]}  ########")

    path = 'tests/dist_mat/'

    # TSKMedoids EUCLIDEAN
    ##########################
    print("\n\n TSKMedoids EUCLIDEAN\n")

    savehere = 'results/NORM/Approach_1/TSKMedoids/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl_gap'

    optimalk = OptimalK_v2(clusterer=kmedoids_euclidean)

    start = datetime.datetime.now()
    n_clusters = optimalk(
        ap_1, 
        n_refs=n_refs, 
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")


    """
    # TSKMedoids DTW
    ##########################
    print("\n\n TSKMedoids DTW\n")

    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
    savehere = 'results/NORM/Approach_1/TSKMedoids/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_dtw_gap'

    optimalk = OptimalK_v2(clusterer=kmedoids_precomputed)

    start = datetime.datetime.now()
    n_clusters = optimalk(
        dist_matrix, 
        n_refs=n_refs, 
        cluster_array=range(2, 51),
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")

    """

    # TSKMedoids DTW w/ Sakoe Chiba r=2
    ##########################
    print("\n\n TSKMedoids DTW with Sakoe Chiba r=2\n")

    dist_matrix = load(path+'dist_mat_dtw_sakoe2_'+agg_app_norm[i][0]+'_norm.npy')
    savehere = 'results/NORM/Approach_1/TSKMedoids/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_dtw_sakoe_2_gap'

    optimalk = OptimalK_v2(clusterer=kmedoids_precomputed)

    start = datetime.datetime.now()
    n_clusters = optimalk(
        dist_matrix, 
        n_refs=n_refs, 
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")


    """
    # TSKMedoids Soft-DTW
    ##########################
    print("\n\TSKMedoids Soft-DTW\n")

    dist_matrix = load(path+'dist_mat_soft_dtw_'+agg_app_norm[i][0]+'_norm.npy')
    savehere = 'results/NORM/Approach_1/TSKMedoids/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_soft_dtw_gap'

    optimalk = OptimalK_v2(clusterer=kmedoids_precomputed)

    start = datetime.datetime.now()
    n_clusters = optimalk(
        dist_matrix, 
        n_refs=n_refs, 
        cluster_array=range(2, 51), 
        save=savehere)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    optimalk.plot_results(save_img=True, img_path=savehere)
    print(f"Number of clusters: {n_clusters}")
    """

#%%
"""
# APROACH 2 #
print("\n\n######### APROACH 2 - GAP STATISTIC #########")

ap_2 = agg_app_norm[i][2]

print("\nKMEANS")
savehere = 'results/Approach_2/KMeans/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl_gap'

optimalk = OptimalK_v2(clusterer=kmeans_euclidean)

start = datetime.datetime.now()
n_clusters = optimalk(
    ap_2, 
    n_refs=n_refs, 
    cluster_array=range(2, 51), 
    save=savehere)
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

optimalk.plot_results(save_img=True, img_path=savehere)
print(f"Number of clusters: {n_clusters}")


print("\nKMEDOIDS")
savehere = 'results/Approach_2/KMedoids/'+agg_app_norm[i][0]+'/'+agg_app_norm[i][0]+'_eucl_gap'

optimalk = OptimalK_v2(clusterer=kmedoids_euclidean)

start = datetime.datetime.now()
n_clusters = optimalk(
    ap_2, 
    n_refs=n_refs, 
    cluster_array=range(2, 51), 
    save=savehere)
stop = datetime.datetime.now()
elapsed_time = str(stop - start)
print(f"time elapsed: {elapsed_time}")

optimalk.plot_results(save_img=True, img_path=savehere)
print(f"Number of clusters: {n_clusters}")

"""

'''
#%%
import numpy as np

# Gap Statistic for K means
def gap_statistic(data, nrefs=100, maxClusters=50):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)

    SOURCE: https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k, n_init=1)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp# Fit cluster to original data and create dispersion
        
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_
        
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
        
    return (gaps.argmax() + 1, resultsdf)


# %%
k, gap_df = gap_statistic(agg_app_norm[i][2], nrefs=n_refs, maxClusters=50)
# %%
k
# %%
gap_df
# %%
'''
