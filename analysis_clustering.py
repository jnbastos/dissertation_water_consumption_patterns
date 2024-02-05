#%%
import time
from tkinter import font
from utils_clustering import *
import seaborn as sns; sns.set()  # for plot styling
import pickle
import numpy as np

#%%
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

df_raw_pkl = pd.read_pickle("Data/dfnotnor.pkl")

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


#%%
################################################################################
################################################################################
def centroids(
        ts, 
        n_clusters, 
        random_state=RANDOM_SEED, 
        xlabel='Hora',
        ylabel='',
        title='', 
        save='', 
        **kwargs):

    start = datetime.datetime.now()
    km = TimeSeriesKMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    
    y_pred = km.fit_predict(ts)
    stop = datetime.datetime.now()
    elapsed_time = str(stop - start)
    print(f"\n\ntime elapsed: {elapsed_time}")


    centroids = km.cluster_centers_

    print(f'Centroids shape {centroids.shape}')
    print("MIN  MAX")
    print(centroids.min(), centroids.max())

    plt.figure()
    for c in range(n_clusters):
        plt.plot(centroids[c].ravel(), alpha=0.8, label='Cluster '+str(c))
    plt.xlim(0, 24)
    plt.ylim(0,centroids.max()*1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    print(title)
    if title != '':
        plt.title(title)
    else:
        plt.title('Centroids')

    plt.legend()

    if save != '':
        plt.savefig(save, dpi=500)

################################################################################
#%%
i = 3



#%%

centroids(
    agg_app_norm[i][1], 
    3, 
    metric="euclidean",
    random_state=RANDOM_SEED,
    ylabel='Consumo (normalizado)',
    title='Dados normalizados',
    save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_euclidean_norm.png'
    )


centroids(
    agg_app_raw[i][1], 
    3, 
    metric="euclidean",
    random_state=RANDOM_SEED,
    ylabel='Consumo (Litros)',
    title='Dados não normalizados',
    save='results/clustering_analysis/centroids/' + agg_app_raw[i][0] + '_euclidean_raw.png'
    )

#%%
print("## Approach 1 ##")
approach_path = 'results/clustering_analysis/'

ts_viz_conf = {
    'barycenter'    : True,
    'average'       : True,
    'dynamic_limits': True,
    'text'          : False,
    'save'          : True,
}

#%%
#* Time Series KMeans *#
path = approach_path + 'results_discussion/norm/'

# TimeSeriesKMeans - Euclidean
print("** TimeSeriesKMeans - Euclidean")

ts_viz_conf['title'] = 'K-Means: Euclidean'

wms_distribution(
    df_norm_pkl,
    agg_app_norm[i][1],
    n_clusters=3,
    clust_func=TimeSeriesKMeans,
    metric="euclidean",
    approach=agg_app_norm[i][0],
    path=path,
    clst_func_str='euclidean',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)

#%%
path = approach_path + 'results_discussion/raw/'

# TimeSeriesKMeans - Euclidean
print("RAW")

ts_viz_conf['title'] = 'K-Means: Euclidean'

wms_distribution(
    df_raw_pkl,
    agg_app_raw[i][1],
    n_clusters=3,
    clust_func=TimeSeriesKMeans,
    metric="euclidean",
    approach=agg_app_raw[i][0],
    path=path,
    clst_func_str='euclidean',
    save_dist_txt=True,
    ts_viz=True,
    ts_viz_conf=ts_viz_conf,
)


#%%
agg_app_raw[i][2]

#%%
#Local 847739; Mes 3; Dia_da_semana 6; Dia 79
# 20/03/2021 Sábado
r1 = df_raw_pkl.loc[6272]['Time series']
n1 = df_norm_pkl.loc[6272]['Time series']

#%%
#Local 847739; Mes 4; Dia_da_semana 5; Dia 99
# 09/04/2021 Sexta
r2 = df_raw_pkl.loc[6292]['Time series']
n2 = df_norm_pkl.loc[6292]['Time series']

#%%
#Local 847739; Mes 5; Dia_da_semana 4; Dia 147
# 27/05/2021 Quinta
r3 = df_raw_pkl.loc[6340]['Time series']
n3 = df_norm_pkl.loc[6340]['Time series']

#%%
#Local 847739; Mes 9; Dia_da_semana 3; Dia 258
# 15/09/2021 Quarta
r4 = df_raw_pkl.loc[6451]['Time series']
n4 = df_norm_pkl.loc[6451]['Time series']


#%%
plt.figure()
plt.plot(r1.ravel(), alpha=0.8, label='20/03/2021')
plt.plot(r2.ravel(), alpha=0.8, label='09/04/2021')
plt.plot(r3.ravel(), alpha=0.8, label='27/05/2021')
plt.plot(r4.ravel(), alpha=0.8, label='15/09/2021')


plt.xlim(0, 24)
#plt.ylim(0,centroids.max()*1.2)
plt.xlabel('Horas')
plt.ylabel('Consumo')
plt.legend()


#%%
#%%
plt.figure()
plt.plot(n1.ravel(), alpha=0.8, label='20/03/2021')
plt.plot(n2.ravel(), alpha=0.8, label='09/04/2021')
plt.plot(n3.ravel(), alpha=0.8, label='27/05/2021')
plt.plot(n4.ravel(), alpha=0.8, label='15/09/2021')


plt.xlim(0, 24)
#plt.ylim(0,centroids.max()*1.2)
plt.xlabel('Horas')
plt.ylabel('Consumo')
plt.legend()



#%%
df_raw_pkl.loc[df_raw_pkl['Dia']==79]

#%%
#807, 1171, 123139, 124232

#%%
#Local 826782; Mes 3; Dia_da_semana 6; Dia 79
# 20/03/2021 Sábado
r1 = df_raw_pkl.loc[807]['Time series']
n1 = df_norm_pkl.loc[807]['Time series']

#%%
#Local 829145; Mes 3; Dia_da_semana 6; Dia 79
# 20/03/2021 Sábado
r2 = df_raw_pkl.loc[1171]['Time series']
n2 = df_norm_pkl.loc[1171]['Time series']

#%%
#Local 4408250; Mes 3; Dia_da_semana 6; Dia 79
# 20/03/2021 Sábado
r3 = df_raw_pkl.loc[123139]['Time series']
n3 = df_norm_pkl.loc[123139]['Time series']

#%%
#Local 4929462; Mes 3; Dia_da_semana 6; Dia 79
# 20/03/2021 Sábado
r4 = df_raw_pkl.loc[124232]['Time series']
n4 = df_norm_pkl.loc[124232]['Time series']

#%%

#%%
plt.figure()
plt.plot(r1.ravel(), alpha=0.8, label='Local 1')
plt.plot(r2.ravel(), alpha=0.8, label='Local 2')
#plt.plot(r3.ravel(), alpha=0.8, label='Local 3')
#plt.plot(r4.ravel(), alpha=0.8, label='15/09/2021')

plt.title('Consumo não normalizado (Dia 20/03/2021)')
plt.xlim(0, 24)
#plt.ylim(0,centroids.max()*1.2)
plt.xlabel('Horas')
plt.ylabel('Consumo (Litros)')
plt.legend()

save = 'results/clustering_analysis/results_discussion/raw/raw_20_03_2021_3locals.png'
plt.savefig(save, dpi=500)

#%%
plt.figure()
plt.plot(n1.ravel(), alpha=0.8, label='Local 1')
plt.plot(n2.ravel(), alpha=0.8, label='Local 2')
#plt.plot(n3.ravel(), alpha=0.8, label='Local 3')
#plt.plot(n4.ravel(), alpha=0.8, label='15/09/2021')

plt.title('Consumo normalizado (Dia 20/03/2021)')
plt.xlim(0, 24)
#plt.ylim(0,centroids.max()*1.2)
plt.xlabel('Horas')
plt.ylabel('Consumo (normalizado)')
plt.legend()

save = 'results/clustering_analysis/results_discussion/norm/norm_20_03_2021_3locals.png'
plt.savefig(save, dpi=500)


#%%
centroids(
    agg_app_norm[i][1], 
    3, 
    metric="euclidean",
    random_state=RANDOM_SEED,
    ylabel='Consumo',
    title='Distância Euclidiana',
    save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_euclidean_norm.png'
    )

centroids(
    agg_app_norm[i][1], 
    3, 
    metric="dtw",
    random_state=RANDOM_SEED,
    ylabel='Consumo',
    title='DTW',
    save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_dtw_norm.png'
    )

centroids(
    agg_app_norm[i][1], 
    3,
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    ylabel='Consumo',
    title='DTW com restrição Sakoe Chiba (r=2)',
    save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_dtwSC2_norm.png'
    )

centroids(
    agg_app_norm[i][1], 
    3, 
    metric="softdtw",
    metric_params={"gamma": 0.03942207089736752},
    random_state=RANDOM_SEED,
    ylabel='Consumo',
    title='Soft-DTW (gamma=0.039)',
    save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_softdtw_norm.png'
    )

#%%
for i in [0,1,2,5]:
    centroids(
        agg_app_norm[i][1], 
        3, 
        metric="euclidean",
        random_state=RANDOM_SEED,
        ylabel='Consumo',
        title='Distância Euclidiana',
        save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_euclidean_norm.png'
        )

    centroids(
        agg_app_norm[i][1], 
        3, 
        metric="dtw",
        random_state=RANDOM_SEED,
        ylabel='Consumo',
        title='DTW',
        save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_dtw_norm.png'
        )

    centroids(
        agg_app_norm[i][1], 
        3,
        metric="dtw",
        metric_params={'sakoe_chiba_radius': 2},
        random_state=RANDOM_SEED,
        ylabel='Consumo',
        title='DTW com restrição Sakoe Chiba (r=2)',
        save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_dtwSC2_norm.png'
        )



#%%
for i in [0,1,2,5]:
    centroids(
        agg_app_norm[i][1], 
        3, 
        metric="softdtw",
        metric_params={"gamma": 0.03942207089736752},
        random_state=RANDOM_SEED,
        ylabel='Consumo',
        title='Soft-DTW (gamma=0.039)',
        save='results/clustering_analysis/centroids/' + agg_app_norm[i][0] + '_softdtw_norm.png'
        )

#%%
################################################################################
#%%
centroids(
    agg_app[i][1], 
    3, 
    metric="dtw",  
    random_state=RANDOM_SEED,
    title='Centroids using DTW',
    save='results/clustering_analysis/centroids/' + agg_app[i][0] + '_dtw.png'
    )

#%%
centroids(
    agg_app[i][1], 
    3, 
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='Centroids using DTW - Sakoe Chiba r=2',
    save='results/clustering_analysis/centroids/' + agg_app[i][0] + '_dtw_sakoe2.png'
    )

#%%
centroids(
    agg_app[i][1], 
    2, 
    metric="softdtw",  
    random_state=RANDOM_SEED,
    title='Centroids using Soft-DTW',
    save='results/clustering_analysis/centroids/' + agg_app[i][0] + '_softdtw.png'
    )


################################################################################



#%%
centroids(
    ts_gdc_norm, 
    3, 
    metric="dtw",
    metric_params={'itakura_max_slope': 6},
    random_state=RANDOM_SEED,
    title='Centroids using DTW - Itakura s=6',
    save='tests/clustering/centroids/centroids_GDC_dtw_Itk6.png'
    )

#%%

################################################################################
################################################################################
#%%
def all_ts_barycenter_viz(
    ts, 
    n_clusters,
    title='', 
    save='',
    average=False,
    barycenter=False,
    dynamic_limits=False,
    text=False,
    **kwargs,
    ):
    km = TimeSeriesKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, **kwargs)
    y_pred = km.fit_predict(ts)
    centroids = km.cluster_centers_


    unique, counts = np.unique(y_pred, return_counts=True)
    print("Distribution of cluster elements => ", str(dict(zip(unique, counts))))

    # Centroids
    # Cluster centroids are calculated by taking the mean of daily
    # profiles that belongs to the assigned cluster number
    average_centroid=[]
    for k in range(n_clusters):
        a1_clust_match = ts[y_pred == k]
        average_centroid.append(sum(a1_clust_match)/len(a1_clust_match))
    print('AVERAGE_CENTROIDS:')
    print(average_centroid)

    total_series = len(ts)
    
    for yi in range(n_clusters):
        plt.figure()
        #plt.subplot(n_clusters, 1, yi + 1)
        series = 0
        
        for xx in ts[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3, linewidth=0.5)
            series += 1
        
        if average:
            plt.plot(average_centroid[yi], "red", linewidth=2, label='média')
        if barycenter:
            plt.plot(centroids[yi].ravel(), "orange", linewidth=2, label='centróide')

        plt.xlim(0, ts.shape[1])

        if dynamic_limits:
            plt.ylim(-0.01,ts[y_pred == yi].max()*1.2)
        else:
            plt.ylim(-0.01,ts.max()*1.2)

        plt.xticks(range(0, 24, 2))

        if text:
            txt = plt.text(1, 0.108, f'{series} de {total_series} séries temporais', fontsize = 14)
            txt.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
        
        plt.legend()

        if title != '':
            plt.title(title + " - Cluster " + str(yi))
        else:
            plt.title("Cluster " + str(yi))

        if save != '':
            plt.savefig(save+'_'+str(yi)+'.png', dpi=250)


#%%
all_ts_barycenter_viz(
    ts_gdc_norm, 
    3, 
    metric="euclidean",
    random_state=RANDOM_SEED,
    title='Distância Euclidiana',
    save='tests/clustering/ts_avg_bary_euclidean',
    barycenter=True,
    average=True
    )

#%%

all_ts_barycenter_viz(
    ts_gdc_norm, 
    3, 
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='DTW (Sakoe Chiba r=2)',
    save='tests/clustering/ts_avg_bary_dtw_r2',
    barycenter=True,
    average=True,
    )

#%%

all_ts_barycenter_viz(
    ts_gdc_norm, 
    3, 
    metric="dtw",
    random_state=RANDOM_SEED,
    title='DTW',
    save='tests/clustering/ts_avg_bary_dtw',
    barycenter=True,
    average=True,
    text=True,
    )


#%%

################################################################################
################################################################################
#%%
# Cluster comparison on different methods
# Euclidean vs DTW Sakoe Chiba r=2

def method_comparison(
    ts,
    n_clusters,
    metric_1="euclidean",
    metric_params_1=None,
    metric_2="dtw",
    metric_params_2=None,
    random_state=RANDOM_SEED,
    title='',
    save='',
    average=False,
    barycenter=False,
    match_text=False,
    match_highlight=False,
    match_heatmap=False,
    **kwargs,
):
    # Clustering with method 1
    km1 = TimeSeriesKMeans(
        n_clusters=n_clusters, 
        metric=metric_1, 
        metric_params=metric_params_1,
        random_state=random_state,
        **kwargs
    )
    y_pred_1 = km1.fit_predict(ts)
    centroids_1 = km1.cluster_centers_

    unique, counts = np.unique(y_pred_1, return_counts=True)
    print("Distribution [method 1] => ", str(dict(zip(unique, counts))))

    # Clustering with method 2
    km2 = TimeSeriesKMeans(
        n_clusters=n_clusters, 
        metric=metric_2, 
        metric_params=metric_params_2,
        random_state=random_state,
        **kwargs
    )
    y_pred_2 = km2.fit_predict(ts)
    centroids_2 = km2.cluster_centers_

    unique, counts = np.unique(y_pred_2, return_counts=True)
    print("Distribution [method 2] => ", str(dict(zip(unique, counts))))

    

    df = pd.DataFrame(data={'y1': y_pred_1, 'y2': y_pred_2})

    # Plotting
    # Centroids
    # Cluster centroids are calculated by taking the mean of daily
    # profiles that belongs to the assigned cluster number
    
    i = 0
    methods = [metric_1, metric_2]
    for idx_, (y_pred, centroids) in enumerate([(y_pred_1, centroids_1), (y_pred_2, centroids_2)]):
        
        average_centroid=[]
        for k in range(n_clusters):
            a1_clust_match = ts[y_pred == k]
            average_centroid.append(sum(a1_clust_match)/len(a1_clust_match))

        total_series = len(ts)
        
        for yi in range(n_clusters):
            plt.figure()
            #plt.subplot(n_clusters, 1, yi + 1)
            series = 0
            n_matches = 0
            
            for xx in ts[y_pred == yi]:
                #plt.plot(xx.ravel(), "k-", alpha=.9, linewidth=0.5)
                plt.plot(xx.ravel(), "yellow", alpha=.5, linewidth=0.4)

                series += 1
            
            if match_highlight:
                for idx in df[(y_pred_1 == yi) & (y_pred_2 == yi)].index:
                    plt.plot(ts[idx].ravel(), "black", alpha=0.7, linewidth=0.5)
                    n_matches += 1

            if average:
                plt.plot(average_centroid[yi], "red", linewidth=2, label='média')
            if barycenter:
                plt.plot(centroids[yi].ravel(), "orange", linewidth=2, label='centróide')

            plt.xlim(0, ts.shape[1])
            plt.ylim(-0.01,ts.max()*1.3)

            plt.xticks(range(0, 24, 2))

            txt = plt.text(1, 0.108, f'cluster com {series} de {total_series} séries temporais', fontsize = 14)
            txt.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))

            if match_text:
                match_of = 0
                if idx_ == 0:
                    match_of = len(df[y_pred_2 == yi])
                if idx_ == 1:
                    match_of = len(df[y_pred_1 == yi])
                txt1 = plt.text(1, 0.095, f'{n_matches} coincidentes', fontsize = 14)
                txt1.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
            
            plt.legend()

            if title != '':
                plt.title(title + " | Cluster " + str(yi) + " | " + methods[i])
            else:
                plt.title("Cluster " + str(yi) + " | Method " + methods[i])

            if save != '':
                plt.savefig(save+'_'+str(yi)+'_' + methods[i] + '.png', dpi=250)

        i+=1

    plt.clf()

    if match_heatmap:
        matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                matrix[i][j] = len(df[(y_pred_1 == i) & (y_pred_2 == j)])
        ax = sns.heatmap(matrix, annot=True, linewidth=.5, fmt='g')
        ax.set(xlabel="Method 2", ylabel="Method 1")
        if save != '':
            plt.savefig(save+'_heatmap.png', dpi=250)

    return y_pred_1, y_pred_2, centroids_1, centroids_2

#%%

## coincidentes em vez de correspondências!
y1, y2, c1, c2 = method_comparison(
    ts=ts_gdc_norm, 
    n_clusters=2,
    metric_1='euclidean',
    metric_2='dtw',
    metric_params_2={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='GDC',
    save='results/clustering_analysis/GDC/EUCLvsDTWsakoe2',
    average=True,
    barycenter=True,
    match_text=True,
    match_highlight=True,
    match_heatmap=True
    )


## coincidentes em vez de correspondências!
y1, y2, c1, c2 = method_comparison(
    ts=ts_gdc_norm, 
    n_clusters=2,
    metric_1='dtw',
    metric_2='dtw',
    metric_params_2={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='GDC',
    save='results/clustering_analysis/GDC/DTWvsDTWsakoe2',
    average=True,
    barycenter=True,
    match_text=True,
    match_highlight=True,
    match_heatmap=True
    )

#%%
y1, y2, c1, c2 = method_comparison(
    ts=ts_gdc_norm, 
    n_clusters=2,
    metric_1='softdtw',
    metric_2='dtw',
    metric_params_2={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='GDC',
    save='results/clustering_analysis/GDC/SOFTDTWvsDTWsakoe2',
    average=True,
    barycenter=True,
    match_text=True,
    match_highlight=True,
    match_heatmap=True
    )

#%%

y1, y2, c1, c2 = method_comparison(
    ts=ts_hwdc_norm, 
    n_clusters=3,
    metric_1='euclidean',
    metric_2='dtw',
    metric_params_2={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='HWDC',
    save='results/clustering_analysis/HWDC/EUCLvsDTWsakoe2',
    average=True,
    barycenter=True,
    match_text=True,
    match_highlight=True,
    match_heatmap=True
    )


## coincidentes em vez de correspondências!
y1, y2, c1, c2 = method_comparison(
    ts=ts_hwdc_norm, 
    n_clusters=3,
    metric_1='dtw',
    metric_2='dtw',
    metric_params_2={'sakoe_chiba_radius': 2},
    random_state=RANDOM_SEED,
    title='HWDC',
    save='results/clustering_analysis/HWDC/DTWvsDTWsakoe2',
    average=True,
    barycenter=True,
    match_text=True,
    match_highlight=True,
    match_heatmap=True
    )




#%%
###################################################
# Days of week and Months distribution by cluster
#
# Clustering approach 2
# Aggregation approach GDC
#
# Dataset Normalized
##################################################
n_clusters = 2

week_month_hist(
    df_norm, 
    df_gdc_norm, 
    n_clusters, 
    save='img/norm/clustering_analysis/'
)


#%%
# GDC - All time series and its mean - Approach 2 clustering
n_clusters = 2

A2_clustering_A1_view(
    df_gdc_norm, 
    ts_gdc_norm, 
    n_clusters, 
    save='img/norm/clustering_analysis/gdc_all_ts',
)

#%%
# n_clusters of each aggregation approach 
Ks = [2,2,2,2]

#%%
for i in range(len(agg_app)):
    print('\n'+agg_app[i][0])
    A2_clust_mean_values_cluster_centroids_A1(
        agg_app[i][2], 
        agg_app[i][1], 
        Ks[i], 
        save=
            'img/norm/clustering_analysis/' +
            agg_app[i][0] +
            '_A2_avg_A1_centroids',
    )
    print('\n')


#%%
######################################
### A1 Clustering - Real Centroids ###
######################################

# TimeSeriesKMeans
for i in range(len(agg_app)):
    print('\n'+agg_app[i][0])
    TimeSeriesKMeans_centroids(
        agg_app[i][1], 
        Ks[i], 
        save=
            'img/norm/clustering_analysis/' +
            agg_app[i][0] +
            '_A1_TSKMeans_real_centroids.png',
    )
    print('\n')


#%%
# KMeans
for i in range(len(agg_app)):
    print('\n'+agg_app[i][0])
    KMeans_centroids(
        agg_app[i][1].squeeze(), 
        Ks[i], 
        save=
            'img/norm/clustering_analysis/'+
            agg_app[i][0]+
            '_A1_KMeans_real_centroids.png',
    )
    print('\n')



#%%
######################################
##              TESTES              ##
######################################

i = 0
centroids_plot_v2(
    agg_app[i][1], 
    Ks[i],
    'img/norm/clustering_analysis/' + agg_app[i][0] + '_centroids_TEST.png',
    TimeSeriesKMeans,
    n_clusters=Ks[i], 
    metric="euclidean", 
    max_iter=10,
    random_state=RANDOM_SEED,
    )

#%%
centroids_plot_v2(
    agg_app[i][1].squeeze(), 
    Ks[i],
    'img/norm/clustering_analysis/' + agg_app[i][0] + '_centroids_TEST_2.png',
    KMeans,
    n_clusters=Ks[i], 
    random_state=RANDOM_SEED,
    )


#%%
Ks[i]


#%%
###################################################
# Centroids
###################################################

'''
n_clusters = 2

# A2 Clustering - Average as centroids
mean_plot(ts_hdc_norm, df_hdc_norm, n_clusters, save='img/norm/A2_centroids')

'''
#%%


# %%
km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
y_pred = km.fit_predict(df_gdc_norm)

centroids = km.cluster_centers_
print(f'Centroids shape {centroids.shape}')     # = (n_clust,5)
print(f'Centroids Approach 2:\n{centroids}')

unique, counts = np.unique(y_pred, return_counts=True)
print("Distribution of cluster elements => ", str(dict(zip(unique, counts))))

for k in range(n_clusters):
    print(f'\nCluster {str(k)} mean values:')
    print(a2_df[y_pred == k].mean())



#%%
## JOCLAD ##

centroids(
    ts_hwdc_norm, 
    3, 
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    max_iter=10,
    title='Centroids using DTW - Sakoe Chiba r=2',
    save='JOCLAD/centroids_HWDC_dtw_SCr2.png'
    )

#%%
all_ts_barycenter_viz(
    ts_hwdc_norm, 
    3, 
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    max_iter=10,
    title='DTW (Sakoe Chiba r=2)',
    save='JOCLAD/ts_avg_bary_dtw_r2',
    barycenter=True,
    average=True,
    text=True,
    )
# %%
all_ts_barycenter_viz(
    ts_hwdc_norm, 
    3, 
    metric="dtw",
    metric_params={'sakoe_chiba_radius': 2},
    title='DTW (Sakoe Chiba r=2)',
    max_iter=10,
    save='JOCLAD/ts_avg_bary_dtw_r2_dynlimits_notext',
    barycenter=True,
    average=True,
    dynamic_limits=True,
    text=False,
    )
# %%
week_month_hist_TS(
    df_norm, 
    ts_gdc_norm, 
    2,
    save='JOCLAD/'
)

## confirmar se resultados fazem entido e ver codigo

#%%
# Check if the Local column is monotonic, i.e. if it is sorted in ascending order
# > True
df_norm.Local.is_monotonic

#%%
df_norm_hwdc = df_norm.sort_values(by=['Local', 'Dia_da_semana'])
df_norm_hwdc.drop_duplicates(subset=['Local', 'Dia_da_semana'], inplace=True)
df_norm_hwdc

#%%
# Check if all Locals have 7 days
# > True
df_norm_hwdc.groupby('Local').count().Dia_da_semana.value_counts()

#%%
df_norm_hwdc.shape, ts_hwdc_norm.shape

#%%
df_aux = df_norm_hwdc[['Dia','Dia_da_semana','Mes']]
df_aux
#%%
k = 3
mode='count'

# Run the kmeans algorithm
km = TimeSeriesKMeans(
                n_clusters=k, 
                metric="dtw",
                metric_params={'sakoe_chiba_radius': 2},
                max_iter=10,
                random_state=RANDOM_SEED)

km.fit(ts_hwdc_norm)
labels = km.predict(ts_hwdc_norm)

#%%
df_aux['Cluster'] = labels.tolist()

#%%
###################################################

df_aux['Cluster'].value_counts(dropna=False)


#%%
pd.Series(labels).value_counts(dropna=False)

#%%
df_aux

#%%
labels.shape

#%%
unique, counts = np.unique(labels, return_counts=True)
print(np.asarray((unique, counts)).T)


###################################################
#%%
n_clusters = 3

# Days of week => Sun = 0, Sat = 6
week_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
for c in range(n_clusters):
    clst_df = df_aux[df_aux['Cluster'] == c]
    print(f"Cluster {str(c)} => Dim = {len(clst_df)} of {len(df_aux)} ({(len(clst_df)/len(df_aux))*100}%)")
    if mode == 'perc':
        day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index()
        stat = 'percent'
        ylabel = "Percentage (%)"
    else:
        day_counts = ((clst_df['Dia_da_semana'].value_counts())).sort_index()
        stat = 'count'
        ylabel = "No. of observations"
    #day_counts = ((clst_df['Dia_da_semana'].value_counts(normalize=True))*100).sort_index() 
    print((day_counts))
    
    plt.figure(figsize=(10,5))
    ax = sns.histplot(data=clst_df, x="Dia_da_semana", stat=stat, discrete=True, shrink=.8)
    plt.title("Cluster " + str(c))
    plt.xlabel("Day of Week")
    plt.xticks(range(7), week_days, rotation='horizontal')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)
    plt.ylabel(ylabel)

    if mode == 'perc':
        plt.ylim(0, 100)
    else:
        pass

    plt.show()
    #plt.savefig(save+'day_of_week_cluster'+str(c)+'.png', dpi=250)
# %%
