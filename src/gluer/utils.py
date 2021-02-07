import seaborn as sns
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from random import sample
import random
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from keras.utils import np_utils
import numba
import warnings
from sklearn.utils import resample

from sklearn import preprocessing


def getDF(obj, data="norm_data"):

    a = pd.DataFrame.sparse.from_spmatrix(obj.layers[data])
    a.columns = obj.var.index
    a.index = obj.obs.index

    return a


def NmaxIndex(test_list, N):

    return sorted(range(len(test_list)), key=lambda sub: test_list[sub])[-N:]


def queryN(rna_atac_metric, i, N):

    return NmaxIndex(list(rna_atac_metric.iloc[:, i].values), N)


def extract_dict(a_dictionary, keys_to_extract):

    return {key: a_dictionary[key] for key in keys_to_extract}


def getQuery(rna_atac_metric, i, N):

    return queryN(rna_atac_metric, i, N)


def getQuery_parallel(rna_atac_metric, i, N):

    return [getQuery(rna_atac_metric, j, N) for j in i]


def query_helper(args):

    return getQuery_parallel(*args)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flattenListOfLists2(lst):
    result = []
    [result.extend(sublist) for sublist in lst]  # uggly side effect ;)
    return result


def parallel_query(rna_atac_metric, N1, N2, n_jobs=5):

    p = multiprocessing.Pool(n_jobs + 1)
    N_rows, N_cols = rna_atac_metric.shape
    njob1 = math.floor(n_jobs / 2)

    job_args1 = [(rna_atac_metric, i, N1) for i in np.array_split(range(N_cols), n_jobs - njob1)]
    job_args2 = [(rna_atac_metric.T, i, N2) for i in np.array_split(range(N_rows), njob1)]
    job_args = job_args1 + job_args2

    result = p.map(query_helper, job_args)
    result = flattenListOfLists2(result)
    p.close()

    return result


def search_mnn(k_index_1, k_index_2, index):
    mutual_1 = []
    mutual_2 = []
    for index_2 in index:
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2


def parallel_mnn(k_index_1, k_index_2, N_cols, n_jobs=5):

    p = multiprocessing.Pool(n_jobs + 1)
    job_args = [(k_index_1, k_index_2, i) for i in np.array_split(range(N_cols), n_jobs)]
    result = p.map(mnn_helper, job_args)
    res = []
    for i in result:
        res.append(pd.DataFrame(i).T)
    result = pd.concat(res)

    p.close()

    return result


def mnn_helper(args):

    return search_mnn(*args)


def single_query(rna_atac_metric, N1):

    N_rows, N_cols = rna_atac_metric.shape
    result = [getQuery(rna_atac_metric, i, N1) for i in range(N_cols)]

    return result


def parallel_find_mutual_nn(rna_atac_metric, N1=3, N2=3, n_jobs=4):

    N_rows, N_cols = rna_atac_metric.shape
    index1 = rna_atac_metric.index
    index2 = rna_atac_metric.columns

    result = parallel_query(rna_atac_metric, N1, N2, n_jobs=n_jobs)
    k_index_1 = result[:N_cols]
    k_index_2 = result[N_cols:]
    result = parallel_mnn(k_index_1, k_index_2, N_cols, n_jobs=n_jobs)

    return pd.DataFrame(result)


def find_mutual_nn(rna_atac_metric, N1=3, N2=3, n_jobs=1):

    N_rows, N_cols = rna_atac_metric.shape
    index1 = rna_atac_metric.index
    index2 = rna_atac_metric.columns

    if n_jobs == 1:
        k_index_1 = single_query(rna_atac_metric, N1)
        k_index_2 = single_query(rna_atac_metric.T, N2)
    else:
        result = parallel_query(rna_atac_metric, N1, N2, n_jobs=n_jobs)
        k_index_1 = result[:N_cols]
        k_index_2 = result[N_cols:]

    mutual_1 = []
    mutual_2 = []
    for index_2 in range(N_cols):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)

    return mutual_1, mutual_2


def find_mutual_nn1(rna_atac_metric1, rna_atac_metric2, N1=3, N2=3):

    N_rows, N_cols = rna_atac_metric1.shape
    index1 = rna_atac_metric1.index
    index2 = rna_atac_metric1.columns

    k_index_1 = single_query(rna_atac_metric1, N1)
    k_index_2 = single_query(rna_atac_metric2.T, N2)

    mutual_1 = []
    mutual_2 = []

    for index_2 in range(N_cols):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)

    return mutual_1, mutual_2


def get_k_smallest(arr, k):

    sm1 = np.array([arr[i, np.argpartition(-arr[i, :], k)[k]] for i in range(arr.shape[0])])
    a_index = ((arr - sm1[:, None]) >= 0)
    d = arr * a_index
    e = np.sum(d, axis=1)
    e[e == 0] = 1

    return np.divide(d, e[:, None])


def getAllPairs(pair_index, rna_dict, atac_dict):

    arr = [[0, 0]]
    for i in range(pair_index.shape[0]):
        arr = np.vstack((arr, np.array(np.meshgrid(rna_dict[pair_index.iloc[i, 0]],
                                                   atac_dict[pair_index.iloc[i, 1]])).T.reshape(-1, 2)))

    arr = pd.DataFrame(arr)
    arr = arr.drop(arr.index[0])
    arr = arr.drop_duplicates()

    return arr


def kdist(m, n):
    dist = np.zeros((m.shape[0], n.shape[0]), dtype=np.float32)
    for i in range(m.shape[0]):
        for j in range(n.shape[0]):
            dist[i, j] = np.dot(m[i], n[j])
    return dist


def getID(arr, rna):

    ident_rna = np.array(rna.obs['seurat_clusters'].astype(int).values)
    arr['id'] = ident_rna[arr.iloc[:, 0]]
    arr['id_true'] = ident_rna[arr.iloc[:, 1]]

    return arr


def getAccuracy(pair_index, rna):

    N_rna = len(rna.obs['seurat_clusters'].astype(int).values)
    print('N-rna: ' + str(N_rna))
    pair_index1 = pair_index.iloc[np.logical_and((pair_index.iloc[:, 0] < N_rna).values,
                                                 (pair_index.iloc[:, 1] < N_rna).values), :]

    getID(pair_index1, rna)
    pair_index3 = pair_index1.iloc[:, [0, 1, 2]]
    pair_index3['id1'] = np.array(rna.obs['seurat_clusters'].astype(int).values)[
        pair_index3.iloc[:, 1]]

    return sum(pair_index3.id == pair_index3.id1) / pair_index3.shape[0]


def filterPairs(arr, similarity_rna_atac1, N1=200, N2=200, n_jobs=1):
    N_rows, N_cols = similarity_rna_atac1.shape

    if n_jobs < 2:
        k_index_1 = single_query(similarity_rna_atac1, N1)
        k_index_2 = single_query(similarity_rna_atac1.T, N2)
    else:
        result = parallel_query(similarity_rna_atac1, N1, N2, n_jobs=n_jobs)
        k_index_1 = result[:N_cols]
        k_index_2 = result[N_cols:]

    arr1 = np.array([0, 0])
    for i in range(arr.shape[0]):
        if (arr.iloc[i, 0] in k_index_1[arr.iloc[i, 1]]) and (arr.iloc[i, 1] in k_index_2[arr.iloc[i, 0]]):
            arr1 = np.vstack((arr1, arr.iloc[i, :]))
    arr1 = np.delete(arr1, (0), axis=0)

    return pd.DataFrame(arr1)


def dist_pca(y_all_new, npc=50):

    x = StandardScaler().fit_transform(y_all_new)
    pca = PCA(n_components=npc, random_state=0)
    principalComponents = pca.fit_transform(x)
    similarity_rna_atac = pd.DataFrame(pairwise_distances(principalComponents,
                                                          principalComponents,
                                                          metric='euclidean')).to_numpy()
    return similarity_rna_atac, principalComponents


def calculateIndex(similarity_rna_atac, N_rna):

    similarity_rna_atac3 = similarity_rna_atac[:N_rna, N_rna:]
    similarity_rna_atac4 = similarity_rna_atac[:, N_rna:]
    diag1 = np.diag(similarity_rna_atac3)
    index1 = sum(((similarity_rna_atac4 - diag1) < 0) * 1) / similarity_rna_atac4.shape[0]

    similarity_rna_atac3 = similarity_rna_atac[N_rna:, :N_rna]
    similarity_rna_atac4 = similarity_rna_atac[:N_rna, :].T
    diag1 = np.diag(similarity_rna_atac3)
    index2 = sum(((similarity_rna_atac4 - diag1) < 0) * 1) / similarity_rna_atac4.shape[0]

    index = np.concatenate((index1, index2))

    return index


def calculateIndex3(similarity_rna_atac, N_rna, N_acc):

    N = similarity_rna_atac.shape[0]
    index0 = np.concatenate((range(N_rna), range(N_rna, N_acc)), axis=0).astype('int')
    dist1 = similarity_rna_atac[index0, :]
    dist1 = dist1[:, index0]

    index1 = calculateIndex(dist1, N_rna)
    index0 = np.concatenate((range(N_rna), range((N_rna + N_acc), N)), axis=0).astype('int')

    dist1 = similarity_rna_atac[index0, :]
    dist1 = dist1[:, index0]

    index2 = calculateIndex(dist1, N_rna)
    index = np.concatenate((index1, index2))

    return index


def selectPairs(df, similarity_matrix, N=3):

    weight_df = pd.DataFrame([[row[0], row[1],
                               similarity_matrix.iloc[row[0], row[1]]]
                              for index, row in df.iterrows()])

    g = []
    for i in range(2):
        g.append(weight_df.
                 groupby([i]).
                 apply(lambda x: x.sort_values([2], ascending=True)).
                 reset_index(drop=True).groupby([i]).head(N))

    g1 = pd.concat(g, ignore_index=True).drop_duplicates()

    return pd.concat(g, ignore_index=True).iloc[:, [0, 1]].drop_duplicates(), g1


def getWeight(data, k=50, isweights=False):
    similarity_gluer = pd.DataFrame(pairwise_distances(data,
                                                       data,
                                                       metric='euclidean'))
    N = data.shape[0]
    dist_m = similarity_gluer.to_numpy()
    index_dist = single_query((-1) * pd.DataFrame(dist_m), k)
    weights = np.zeros([N, N])
    if isweights:
        for i in range(N):
            sum_exp = np.exp(-dist_m[index_dist[i], i])
            weights[index_dist[i], i] = sum_exp / sum(sum_exp)
    else:
        for i in range(N):
            weights[index_dist[i], i] = 1 / k

    return weights


def calculateGroupIndex(similarity_seurat,
                        similarity_liger,
                        similarity_gluer,
                        cell_selected):

    similarity_seurat_np = similarity_seurat.loc[cell_selected, cell_selected].to_numpy()
    similarity_liger_np = similarity_liger.loc[cell_selected, cell_selected].to_numpy()
    similarity_gluer_np = similarity_gluer.loc[cell_selected, cell_selected].to_numpy()
    N = int(len(cell_selected) / 2)
    print(N)

    # seurat
    index_seurat = calculateIndex(similarity_seurat_np, N)
    index_seurat = pd.DataFrame(index_seurat)
    index_seurat.index = cell_selected[:len(index_seurat)]
    index_seurat.columns = ["data"]
    index_seurat['method'] = ['Seurat'] * len(index_seurat)
    print(np.mean(index_seurat))

    # liger
    index_liger = calculateIndex(similarity_liger_np, N)
    index_liger = pd.DataFrame(index_liger)
    index_liger.index = cell_selected[:len(index_liger)]
    index_liger.columns = ["data"]
    index_liger['method'] = ['Liger'] * len(index_liger)
    print(np.mean(index_liger))

    # gluer
    index_gluer = calculateIndex(similarity_gluer_np, N)
    index_gluer = pd.DataFrame(index_gluer)
    index_gluer.index = cell_selected[:len(index_gluer)]
    index_gluer.columns = ["data"]
    index_gluer['method'] = ['Gluer'] * len(index_gluer)
    print(np.mean(index_gluer))

    # concatenate all index
    df_index = pd.concat([index_seurat, index_liger, index_gluer])

    return df_index


def _custom_palettes():
    return {
        'YellowOrangeBrown': 'YlOrBr',
        'YellowOrangeRed': 'YlOrRd',
        'OrangeRed': 'OrRd',
        'PurpleRed': 'PuRd',
        'RedPurple': 'RdPu',
        'BluePurple': 'BuPu',
        'GreenBlue': 'GnBu',
        'PurpleBlue': 'PuBu',
        'YellowGreen': 'YlGn',
        'summer': 'summer_r',
        'copper': 'copper_r',
        'viridis': 'viridis_r',
        'cividis': 'cividis_r',
        'plasma': 'plasma_r',
        'inferno': 'inferno_r',
        'magma': 'magma_r',
        'sirocco': sns.cubehelix_palette(
            dark=0.15, light=0.95, as_cmap=True),
        'drifting': sns.cubehelix_palette(
            start=5, rot=0.4, hue=0.8, as_cmap=True),
        'melancholy': sns.cubehelix_palette(
            start=25, rot=0.4, hue=0.8, as_cmap=True),
        'enigma': sns.cubehelix_palette(
            start=2, rot=0.6, gamma=2.0, hue=0.7, dark=0.45, as_cmap=True),
        'eros': sns.cubehelix_palette(start=0, rot=0.4, gamma=2.0, hue=2,
                                      light=0.95, dark=0.5, as_cmap=True),
        'spectre': sns.cubehelix_palette(
            start=1.2, rot=0.4, gamma=2.0, hue=1, dark=0.4, as_cmap=True),
        'ambition': sns.cubehelix_palette(start=2, rot=0.9, gamma=3.0, hue=2,
                                          light=0.9, dark=0.5, as_cmap=True),
        'mysteriousstains': sns.light_palette(
            'baby shit green', input='xkcd', as_cmap=True),
        'daydream': sns.blend_palette(
            ['egg shell', 'dandelion'], input='xkcd', as_cmap=True),
        'solano': sns.blend_palette(
            ['pale gold', 'burnt umber'], input='xkcd', as_cmap=True),
        'navarro': sns.blend_palette(
            ['pale gold', 'sienna', 'pine green'], input='xkcd', as_cmap=True),
        'dandelions': sns.blend_palette(
            ['sage', 'dandelion'], input='xkcd', as_cmap=True),
        'deepblue': sns.blend_palette(
            ['really light blue', 'petrol'], input='xkcd', as_cmap=True),
        'verve': sns.cubehelix_palette(
            start=1.4, rot=0.8, gamma=2.0, hue=1.5, dark=0.4, as_cmap=True),
        'greyscale': sns.blend_palette(
            ['light grey', 'dark grey'], input='xkcd', as_cmap=True)}


def generate_cmap(colors=["lightgray", "blue"],
                  cvals=[0, 1]):
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    return matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


tab10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
tab20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
tab40 = [
    "#393b79",
    "#5254a3",
    "#6b6ecf",
    "#9c9ede",
    "#637939",
    "#8ca252",
    "#b5cf6b",
    "#cedb9c",
    "#8c6d31",
    "#bd9e39",
    "#e7ba52",
    "#e7cb94",
    "#843c39",
    "#ad494a",
    "#d6616b",
    "#e7969c",
    "#7b4173",
    "#a55194",
    "#ce6dbd",
    "#de9ed6",
    "#3182bd",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#e6550d",
    "#fd8d3c",
    "#fdae6b",
    "#fdd0a2",
    "#31a354",
    "#74c476",
    "#a1d99b",
    "#c7e9c0",
    "#756bb1",
    "#9e9ac8",
    "#bcbddc",
    "#dadaeb",
    "#636363",
    "#969696",
    "#bdbdbd",
    "#d9d9d9",
]

# Set3=['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69',
#      '#fccde5','#d9d9d9','#bc80bd']

my_gradients = [
    [[0, "rgb(180,180,180)"], [1, "rgb(31,119,180)"]],  # grey to blue
    [[0, "rgb(180,180,180)"], [1, "rgb(214,39,40)"]],  # grey to red
    [[0, "rgb(180,180,180)"], [1, "rgb(44,160,44)"]],  # grey to green
    [[0, "rgb(180,180,180)"], [1, "rgb(0,0,0)"]],  # grey to black
]


def get_cm(values):
    """Return color mapping"""
    if len(values) < 10:
        return tab10
    elif len(values) < 20:
        return tab20
    else:
        return tab40
