import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from tslearn.clustering import silhouette_score as ts_sil
from tslearn.clustering import TimeSeriesKMeans
from sklearn_extra.cluster import KMedoids

from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import single, complete, average, ward, median, centroid, weighted, dendrogram


RANDOM_SEED = 42

sns.set_style("darkgrid")


from functools import wraps
import time

# Timeit decorator
# https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper




#######################################################
#                  ELBOW METHOD                       #
#######################################################

# conda install -c districtdatalabs yellowbrick
# conda install h5py
# pip install yellowbrick==1.3.post1  para resolver o erro: AttributeError: module 'h5py' has no attribute 'version'
# https://github.com/DistrictDataLabs/yellowbrick/issues/1137
# que consequentemente fez o downgrade do numpy para 1.19.5
#@timeit
def elbow(
    data, 
    k_range, 
    clust_func,
    metric = None,
    metric_yb='distortion', 
    timings=False, 
    locate_elbow=True, 
    save='',
    clear_figure=True,
    **kwargs
    ):
    
    # Instantiate the clustering model and visualizer
    if clust_func == TimeSeriesKMeans:
        model = clust_func(random_state=RANDOM_SEED, metric=metric, **kwargs)
    elif clust_func == KMeans:
        model = clust_func(random_state=RANDOM_SEED, n_init='auto', **kwargs)
    elif clust_func == KMedoids:
        model = clust_func(random_state=RANDOM_SEED, metric = metric, **kwargs)

    visualizer = KElbowVisualizer(model, k=k_range, metric=metric_yb, timings=timings, locate_elbow=locate_elbow)

    visualizer.fit(data)        # Fit the data to the visualizer
    
    if save != '':
        visualizer.show(outpath=save+'.png', clear_figure=True)
    # Finalize and render the figure
    else:
        # clear figure so that it doesn't show up in the next plot
        if clear_figure:
            visualizer.show(clear_figure=True)
        else:
            visualizer.show()

    return visualizer.k_scores_, visualizer.k_timers_, visualizer.elbow_value_, visualizer.elbow_score_        


def elbow_method(dfr, n_clusters, save=''):
    wcss = []

    for k in n_clusters:    
        # Run the kmeans algorithm
        km = KMeans(n_clusters=k,random_state=RANDOM_SEED)
        cluster_labels = km.fit_predict(dfr)
        centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        wcss.append(km.inertia_)

    print(f'\nWithin-Clusters Sum-of-Squares (WCSS):\n{wcss}')

    #View: Elbow Method
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Número de clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Método do cotovelo')
    plt.grid(True)

    y2 = wcss[len(wcss)-1]
    y1 = wcss[0]

    plt.plot([max(n_clusters), min(n_clusters)], [y2,y1], color='darkorange') # reta min-max
    plt.plot(n_clusters, wcss, color='tab:blue') # linha principal
    plt.plot(n_clusters, wcss, '.', color='r') # pontos

    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.xticks(n_clusters)
    #plt.show()

    if save != '':
        plt.savefig(save+'.png', dpi=250)

    plt.clf
    plt.close('all')

    x1, y1 = n_clusters[0], wcss[0]
    x2, y2 = n_clusters[-1], wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    best_k = distances.index(max(distances)) + 2
    print('By Elbow Method the best clustering is with k =', best_k)


def elbow_method_TS(ts, n_clusters, save=''):
    wcss = []

    for k in n_clusters:    
        # Run the kmeans algorithm
        km = TimeSeriesKMeans(
                        n_clusters=k, 
                        metric="euclidean", 
                        max_iter=10,
                        random_state=RANDOM_SEED)
        km.fit(ts)
        cluster_labels = km.predict(ts)
        #cluster_labels = km.fit_predict(ts)
        centroids = km.cluster_centers_
        #print("centroids shape", centroids.shape)

        wcss.append(km.inertia_)

    print(f'\nWithin-Clusters Sum-of-Squares (WCSS):\n{wcss}')

    #View: Elbow Method
    fig = plt.figure()
    ax5 = fig.add_subplot()
    ax5.set_xlabel('Número de clusters')
    ax5.set_ylabel('WCSS')
    ax5.set_title('Método do cotovelo')
    plt.grid(True)

    y2 = wcss[len(wcss)-1]
    y1 = wcss[0]

    plt.plot([max(n_clusters), min(n_clusters)], [y2,y1], color='darkorange') # reta min-max
    plt.plot(n_clusters, wcss, color='tab:blue') # linha principal
    plt.plot(n_clusters, wcss, '.', color='r') # pontos

    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.xticks(n_clusters)
    # plt.show()

    if save != '':
        plt.savefig(save+'.png', dpi=250)
    plt.clf
    plt.close('all')

    x1, y1 = n_clusters[0], wcss[0]
    x2, y2 = n_clusters[-1], wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    best_k = distances.index(max(distances)) + 2
    print('By Elbow Method the best clustering is with k =', best_k)


#######################################################
#                   SILHOUETTE                        #
#######################################################
#@timeit
def silhouette(
    dfr,
    k_range, 
    clust_func=None,
    dist_matrix = None,
    random_state=RANDOM_SEED, 
    metric_params=None, 
    sil_metric=None,
    save_txt='', 
    imgs=False,
    save_img='',
    save_all_results=False,
    A2=False,
    labels_only=False,
    ts=None,
    cluster_labels=None,
    calinski=False,
    **kwargs):
    '''
    Computes the silhouette score for a range of k (number of cluster) values.
    
    Parameters
    ----------
    dfr : data
        Data to be clustered.
    k_range : list
        List of k values to be tested.
    clust_func : function
        Clustering function to be used.
    metric_params : dict
        Parameters to be passed to the clustering function and the silhouette 
        metric function (if TimeSeriesKMeans is used).
    sil_metric : function
        Silhouette metric function to be used. If None, the default metric
        function is used.
    save_img : str
        Path to save the silhouette plot. Only used if imgs is True.
    imgs : bool
        If True, the silhouette plot is shown.
    **kwargs : dict
        Additional parameters to be passed to the clustering function.

    '''
    results = []
    inertia = []
    n_iter = []
    for k in k_range:
        if imgs:
            fig, (ax1) = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)
            ax1.set_xlim([-0.1, 1])
            # The (k+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(dfr) + (k + 1) * 10])
        
        if labels_only:
            if calinski:
                silhouette_avg = calinski_harabasz_score(dfr, cluster_labels[k-2])


            else:
                silhouette_avg = silhouette_score(dfr, cluster_labels[k-2], metric=sil_metric)
                if imgs:
                        # Compute the silhouette scores for each sample
                        sample_silhouette_values = silhouette_samples(dfr, cluster_labels[k-2])

        else:
            # Run the clustering algorithm
            if clust_func == TimeSeriesKMeans:
                km = clust_func(n_clusters=k, random_state=random_state, metric_params=metric_params, **kwargs)
            elif clust_func == KMedoids:
                km = clust_func(n_clusters=k, random_state=random_state, **kwargs)
            elif clust_func == KMeans:
                km = clust_func(n_clusters=k, n_init='auto', random_state=random_state, **kwargs)


            km.fit(dfr)
            cluster_labels = km.predict(dfr)

            # Save inertia and number of iterations for each k
            inertia.append(km.inertia_)
            n_iter.append(km.n_iter_)

            print_inert_iter = f'k={k} inertia={km.inertia_} iterations={km.n_iter_}'
            #print(print_inert_iter)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            if (clust_func == TimeSeriesKMeans) or (clust_func == KMedoids):
                if dist_matrix is not None:
                    silhouette_avg = ts_sil(dist_matrix, cluster_labels, metric=sil_metric, metric_params=metric_params, n_jobs=-1)
                else:
                    # DTW is the default metric for tslearn silhouette
                    silhouette_avg = ts_sil(dfr, cluster_labels, metric=sil_metric, metric_params=metric_params, n_jobs=-1)

                if imgs:
                    # Compute the silhouette scores for each sample
                    sample_silhouette_values = silhouette_samples(dfr.reshape(len(dfr),24), cluster_labels)
            
            #elif A2 and (ts is not None):
            #    print("Silhoette calculated using time series")
            #    # Euclidean is the default metric for sklearn silhouette
            #    #silhouette_avg = silhouette_score(ts, cluster_labels, metric=sil_metric)
            #    silhouette_avg = ts_sil(ts, cluster_labels, metric=sil_metric, n_jobs=-1)
            #
            #    if imgs:
            #        # Compute the silhouette scores for each sample
            #        #sample_silhouette_values = silhouette_samples(ts, cluster_labels)
            #        sample_silhouette_values = silhouette_samples(ts.reshape(len(ts),24), cluster_labels)
                
            elif clust_func == KMeans:
                # Euclidean is the default metric for sklearn silhouette
                silhouette_avg = silhouette_score(dfr, cluster_labels, metric=sil_metric)

                if imgs:
                    # Compute the silhouette scores for each sample
                    sample_silhouette_values = silhouette_samples(dfr, cluster_labels)
        

            
        print_score = f'k = {k} => Average Silhouette Score = {round(silhouette_avg, 4)}'
        if save_txt != '':
            with open(save_txt + '.txt',"a+") as f:
                f.write(print_score + '\n')

            if save_all_results:
                with open(save_txt + '_inertia_niter.txt',"a+") as f:
                    f.write(print_inert_iter + '\n')

        print(print_score)

        results.append(silhouette_avg)

        if imgs:
            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    alpha=0.7,
                )
                
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title(f"The silhouette plot for the various clusters (with  k = {k})")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette = "+str(round(silhouette_avg,2)))
            ax1.legend(loc='upper right')
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            if save_img != '':
                plt.savefig(save_img+'_'+str(k)+'.png', dpi=250)
            else:
                plt.show()

            # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
            plt.clf
            plt.close('all')

    # Print higher silhouette score
    best_score = max(results)
    best_k_idx = results.index(best_score)

    print_best_score = f'\nBest score: {best_score} with k = {k_range[best_k_idx]}\n\n'
    if save_txt != '':
        with open(save_txt + '.txt',"a+") as f:
                f.write(print_best_score)
    print(print_best_score)

    return results, inertia, n_iter


# WCSS ( Within-Cluster Sum of Square ) is the sum of squared distance between 
# each point and the centroid in a cluster.
'''
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2
'''

def hierarchical_clustering(dist_mat, method='complete', save=''):
    start = datetime.datetime.now()
    if method == 'complete':
        Z = complete(dist_mat)
    elif method == 'single':
        Z = single(dist_mat)
    elif method == 'average':
        Z = average(dist_mat)
    elif method == 'ward':
        Z = ward(dist_mat)
    elif method == 'median':
        Z = median(dist_mat)
    elif method == 'centroid':
        Z = centroid(dist_mat)
    elif method == 'weighted':
        Z = weighted(dist_mat)
    
    fig = plt.figure(figsize=(16, 8))
    plt.grid(False)
    dn = dendrogram(Z)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"time elapsed: {elapsed_time}")

    plt.title(f"Dendrogram for {method}-linkage with correlation distance")


    if save != '':
        plt.savefig(save + '_' + method + '.png', dpi=250, transparent=True)
    else:
        plt.show()

    plt.close('all')
    
    return Z