import numpy as np
import pandas as pd
import multiprocessing
import time
from sklearn.metrics import pairwise_distances
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_kernels
import json
from random import sample
import random
from . import iONMF
import sys
import re
import umap
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from keras.utils import np_utils
import numba
from sklearn.utils import resample
from scipy.sparse import csr_matrix
from .utils import *
import os
import pkg_resources


def gluer(ref_obj,
               query_obj,
               joint_rank=20,
               joint_max_iter=200,
               joint_random_seed=21,
               mnn_ref=30,
               mnn_query=30,
               filter_n1=50,
               filter_n2=50,
               N=3,
               n_jobs=1,
               n_features=15000,
               is_impute=True,
               filter_n_features=[15000, 15000],
               pairs=None,
               deep_random_seed=44,
               deepmodel_epoch=500,
               batch_categories=['1', '2'],
               model=None,
               validation_split=.1,
               verbose=0):
    """Short summary.

    Parameters
    ----------
    ref_obj : h5ad file
        The AnnData data object of the reference data.
    query_obj : type
        Description of parameter `query_obj`.
    joint_rank : type
        Description of parameter `joint_rank`.
    joint_max_iter : type
        Description of parameter `joint_max_iter`.
    joint_random_seed : type
        Description of parameter `joint_random_seed`.
    mnn_ref : type
        Description of parameter `mnn_ref`.
    mnn_query : type
        Description of parameter `mnn_query`.
    filter_n1 : type
        Description of parameter `filter_n1`.
    filter_n2 : type
        Description of parameter `filter_n2`.
    N : type
        Description of parameter `N`.
    n_jobs : type
        Description of parameter `n_jobs`.
    n_features : type
        Description of parameter `n_features`.
    is_impute : type
        Description of parameter `is_impute`.
    filter_n_features : type
        Description of parameter `filter_n_features`.
    pairs : type
        Description of parameter `pairs`.
    deep_random_seed : type
        Description of parameter `deep_random_seed`.
    deepmodel_epoch : type
        Description of parameter `deepmodel_epoch`.
    batch_categories : type
        Description of parameter `batch_categories`.
    model : type
        Description of parameter `model`.
    validation_split : type
        Description of parameter `validation_split`.
    verbose : type
        Description of parameter `verbose`.
    query_obj.var.sort_values(by : type
        Description of parameter `query_obj.var.sort_values(by`.
    query_obj.var.sort_values(by : type
        Description of parameter `query_obj.var.sort_values(by`.
     : common_feature].to_numpy()
        Description of parameter ``.
     : common_feature_selected].to_numpy()
        Description of parameter ``.
    common_feature_selected].to_numpy( : type
        Description of parameter `common_feature_selected].to_numpy(`.

    Returns
    -------
    type
        Description of returned object.

    """
    start_time_all = time.time()
    sys.stdout.write("=========================================== Gluer =================================================\n" +
                     "Four steps are as follows:\n" +
                     "Step 1: Jointly dimension reduction model\n" +
                     "Step 2: Search the cell pairs between the reference and the query\n" +
                     "Step 3: Run the deep learning model\n" +
                     "Step 4: Summarize the output\n" +
                     "===================================================================================================\n")
    sys.stdout.flush()

    common_feature = np.intersect1d(ref_obj.var.sort_values(by=['vst_variance_standardized'],
                                                            ascending=False).index.values[:n_features],
                                    query_obj.var.sort_values(by=['vst_variance_standardized'],
                                                              ascending=False).index.values[:n_features])

    common_feature_selected = np.intersect1d(ref_obj.var.sort_values(by=['vst_variance_standardized'],
                                                                     ascending=False).index.values[:filter_n_features[0]],
                                             query_obj.var.sort_values(by=['vst_variance_standardized'],
                                                                       ascending=False).index.values[:filter_n_features[1]])

    data_ref_raw = getDF(ref_obj)
    data_query_raw = getDF(query_obj)

    # prepare the reference data and query data for the integration
    data_ref = data_ref_raw.loc[:, common_feature].to_numpy()
    data_query = [data_query_raw.loc[:, common_feature].to_numpy()]

    data_ref_selected = data_ref_raw.loc[:, common_feature_selected].to_numpy()
    data_query_selected = data_query_raw.loc[:, common_feature_selected].to_numpy()

    if is_impute:
        weights = getWeight(ref_obj.obsm['umap_cell_embeddings'])
        data_ref = np.dot(data_ref.T, weights).T

    # prepare thes dataset for the jointly dimension reduction
    sys.stdout.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                     " >> Step 1: Jointly dimension reduction model ... ")

    start_time = time.time()
    dataset = {'data' + str(i + 1): data.T for i, data in enumerate(data_query)}
    dataset['ref'] = data_ref.T
    # setup the jointly dimension reduction models
    model_joint = iONMF.iONMF(rank=joint_rank,
                              max_iter=joint_max_iter,
                              alpha=1,
                              random_seed=21)

    model_joint.fit(dataset)
    msg = "Done %s mins \n" % round((time.time() - start_time) / 60, 2)
    sys.stdout.write(msg)
    sys.stdout.flush()

    N_ref_obj = data_ref.shape[0]
    # define the list to store the intermediate results
    data_ref_name = "ref"
    data_ref = dataset[data_ref_name].T
    pair_ref_query_list = list()
    model_deepLearning_list = list()
    y_pred_ref_list = list()
    y_pred_ref_list.append(model_joint.basis_[data_ref_name].T)

    for j in range(1, len(dataset)):
        sys.stdout.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                         " >> Step 2-" + str(j) +
                         ": Search the cell pairs ... ")
        data_query_name = "data" + str(j)
        data_query = dataset[data_query_name].T

        if pairs is None:
            # calculate the similarity between reference data and query data
            similarity_ref_query = pd.DataFrame(
                pairwise_kernels(
                    model_joint.basis_[data_ref_name].T,
                    model_joint.basis_[data_query_name].T,
                    metric='cosine')
            )

            # raw similarity
            similarity_selected = pd.DataFrame(
                pairwise_kernels(data_ref_selected,
                                 data_query_selected,
                                 metric='cosine')
            )

            # find out the cell pairs between reference data and query data
            ref_pair, query_pair = find_mutual_nn(similarity_ref_query,
                                                  N1=mnn_ref,
                                                  N2=mnn_query,
                                                  n_jobs=n_jobs)

            pair_ref_query = pd.DataFrame([ref_pair, query_pair]).T

            print("before filtering: " + str(pair_ref_query.shape[0]))

            pair_ref_query = filterPairs(pair_ref_query,
                                         similarity_selected,
                                         N1=filter_n1,
                                         N2=filter_n2,
                                         n_jobs=n_jobs)

            # remove the duplicates in case there is
            pair_ref_query.drop_duplicates()

            pair_ref_query, g1 = selectPairs(pair_ref_query,
                                             similarity_ref_query,
                                             N=N)
        else:
            cell_index = pd.DataFrame(np.arange(N_ref_obj))
            pair_ref_query = pd.concat((cell_index, cell_index), axis=1)

        msg = "found " + str(pair_ref_query.shape[0]) + ' pairs ... '
        sys.stdout.write(msg)
        sys.stdout.flush()

        msg = "Done %s mins \n" % round((time.time() - start_time) / 60, 2)
        sys.stdout.write(msg)
        sys.stdout.flush()
        sys.stdout.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                         " >> Step 3: Run the deep learning model ... ")

        # prepare the deep learning model data
        x_train = model_joint.basis_[data_query_name].T[pair_ref_query.iloc[:, 1].values]
        y_train = model_joint.basis_[data_ref_name].T[pair_ref_query.iloc[:, 0].values]
        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        # train the deep learning model
        if model is None:
            tf.random.set_random_seed(deep_random_seed)
            start_time = time.time()
            model = tf.keras.Sequential()
            model.add(layers.Dense(input_dim, activation='relu'))
            model.add(layers.Dense(200, activation='relu'))
            model.add(layers.Dense(100, activation='relu'))
            model.add(layers.Dense(50, activation='relu'))
            model.add(layers.Dense(25, activation='relu'))
            model.add(layers.Dense(50, activation='relu'))
            model.add(layers.Dense(100, activation='relu'))
            model.add(layers.Dense(200, activation='relu'))
            model.add(layers.Dense(output_dim, activation='relu'))
            model.compile(loss="mean_squared_error",
                          optimizer="adam",
                          metrics=['accuracy'])

        factor_val = 1e8
        history = model.fit(x_train * factor_val,
                            y_train * factor_val,
                            validation_split=validation_split,
                            epochs=deepmodel_epoch,
                            verbose=verbose)
        # predict the reference data based on the query datasets
        y_pred_ref_list.append(model.predict(
            model_joint.basis_[data_query_name].T * factor_val) / factor_val)
        # save all intermediate results
        pair_ref_query_list.append(pair_ref_query)
        model_deepLearning_list.append(model)
        msg = "Done %s mins \n" % round((time.time() - start_time) / 60, 2)
        sys.stdout.write(msg)
        sys.stdout.flush()
    sys.stdout.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                     " >> Step 4: Summarize the output ... ")
    start_time = time.time()
    # prepare AnnData
    y_pred_ref = np.concatenate(y_pred_ref_list, axis=0)
    gdata = ref_obj.concatenate(query_obj,
                                batch_key='gluer_batch',
                                batch_categories=batch_categories,
                                index_unique='_')

    gdata.layers['norm_data'] = csr_matrix(
        pd.concat([data_ref_raw.loc[:, gdata.var.index.values],
                   data_query_raw.loc[:, gdata.var.index.values]]).to_numpy())

    # set up the dimension reduction
#     keys_gdata = gdata.obsm.keys()
#     for k in keys_gdata:
#         kk = re.sub("pca", "pca_raw", k)
#         if k != kk:
#             gdata.obsm[kk] = gdata.obsm[k]
#         kk = re.sub("tsne", "tsne_raw", k)
#         if k != kk:
#             gdata.obsm[kk] = gdata.obsm[k]
#         kk = re.sub("umap", "umap_raw", k)
#         if k != kk:
#             gdata.obsm[kk] = gdata.obsm[k]

    gdata.obsm['igluer'] = y_pred_ref
    gdata.uns['joint_nmf'] = vars(model_joint)
    gdata.uns['dataset'] = dataset
    gdata.uns['history'] = pd.DataFrame(history.history)
    gdata.uns['pairs'] = pair_ref_query.to_numpy()

    parameters = {'joint_rank': joint_rank,
                  'joint_max_iter': joint_max_iter,
                  'joint_random_seed': joint_random_seed,
                  'mnn_ref': mnn_ref,
                  'mnn_query': mnn_query,
                  'filter_n1': filter_n1,
                  'filter_n2': filter_n2,
                  'N': N,
                  'n_jobs': n_jobs,
                  'n_features': n_features,
                  'is_impute': is_impute,
                  'filter_n_features': filter_n_features,
                  'pairs': pairs,
                  'deep_random_seed': deep_random_seed,
                  'deepmodel_epoch': deepmodel_epoch,
                  'batch_categories': batch_categories,
                  'model': model,
                  'validation_split': validation_split,
                  'verbose': verbose}

    gdata.uns['parameter'] = parameters

    msg = "Done %s mins \n" % round((time.time() - start_time) / 60, 2)
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write(
        "==============================================================\
        =====================================\n")
    sys.stdout.flush()
    msg1 = "The whole job is done in %s mins " % \
           round((time.time() - start_time_all) / 60, 2)
    msg2 = "with %s features used in this run" % len(common_feature)
    msg = msg1 + msg2
    sys.stdout.write(msg)
    sys.stdout.flush()

    return gdata


def run_impute(gluer_obj, k=20, isweights=True):

    y_pred_ref = gluer_obj.obsm['igluer']
    N_ref_obj = gluer_obj.uns['dataset']['ref'].shape[1]

    similarity_gluer = pd.DataFrame(pairwise_distances(y_pred_ref,
                                                       y_pred_ref,
                                                       metric='euclidean'))
    N = y_pred_ref.shape[0]
    dist_m = similarity_gluer.to_numpy()[:N_ref_obj, :]
    index_dist = single_query((-1) * pd.DataFrame(dist_m), k)
    weights = np.zeros([N_ref_obj, N])
    if isweights:
        for i in range(N):
            sum_exp = np.exp(-dist_m[index_dist[i], i])
            weights[index_dist[i], i] = sum_exp / sum(sum_exp)
    else:
        for i in range(N):
            weights[index_dist[i], i] = 1 / k

    gluer_obj.layers['norm_data'][:N_ref_obj, :]

    gluer_obj.layers['imputed_data'] = np.dot(
        gluer_obj.layers['norm_data'][:N_ref_obj, :].T, weights).T

    return gluer_obj


def run_umap(gdata,
             n_neighbors=40,
             min_dist=0.1,
             n_components=2,
             metric='cosine'):
    numba.set_num_threads(4)
    mapper = umap.UMAP(n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       n_components=2,
                       metric=metric).fit(gdata.obsm['igluer'])

    gdata.obsm['X_umap'] = mapper.embedding_
    return gdata


def load_demo_data():
    # load the data
    stream = pkg_resources.resource_stream(__name__,
                                           'data/RNA_demo_github.h5ad')
    rna_data = sc.read_h5ad(stream)
    stream = pkg_resources.resource_stream(__name__,
                                           'data/ACC_demo_github.h5ad')
    acc_data = sc.read_h5ad(stream)
    stream = pkg_resources.resource_stream(__name__,
                                           'data/GLUER_demo_github.h5ad')
    gluer_data = sc.read_h5ad(stream)

    return rna_data, acc_data, gluer_data
