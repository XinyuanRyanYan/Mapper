import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import kmapper as km
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import torch.utils.data as torchdata
import pandas as pd
from enhanced_mapper.cover import Cover as enhanced_Cover
from enhanced_mapper.mapper import generate_mapper_graph
from enhanced_mapper.AdaptiveCover import mapper_xmeans_centroid

def iprint(*args, **kwargs):
    print('>', *args, **kwargs)

@dataclass
class NodeStat:
    node_norm: float
    misclassification_rate: float
    class_compositions: List[float]
    num_vertices: int

def create_mapper_graph(activations, intervals, overlap, categorical_cols, output_dir, output_fname, eps=None, is_enhanced_cover=False, enhanced_parameters=None, verbose=0):
    def elbow_eps(data):
        # estimate nearest neighbor using a subset of data
        data_subset = data[np.random.choice(len(data), size=min(len(data), 5000), replace=False)]
        nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(data_subset)
        distances, indices = nbrs.kneighbors(data)
        distances = np.sort(distances[:, 1], axis=0)
        kneedle = kneed.KneeLocator(np.arange(len(distances)), distances, curve='convex',
                                    direction='increasing', interp_method='polynomial')
        eps = distances[kneedle.knee]
        # kneedle.plot_knee()
        return eps

    if not eps:
        eps = elbow_eps(activations)
    iprint(f'eps: {eps}')

    print(activations.shape)

    max_intervals = 100
    if enhanced_parameters!=None:
        iterations = enhanced_parameters['iterations']
        delta = enhanced_parameters['delta']
        method = enhanced_parameters['method'] # ["BFS", "DFS", "randomized"]
        BIC = enhanced_parameters['bic'] # ["BIC, "AIC"]
    else:
        iterations = 100
        delta = 0.1
        method = "BFS"
        BIC = "BIC" 

    projected_activations = np.linalg.norm(activations, axis=1).reshape(-1,1)
    print(projected_activations)

    clusterer=DBSCAN(eps=eps, min_samples=5, n_jobs=-1)

    if is_enhanced_cover:
        cover = enhanced_Cover(intervals, overlap)
        multipass_cover = mapper_xmeans_centroid(activations, projected_activations, cover, clusterer, iterations, max_intervals, BIC=BIC, delta=delta, method=method)
        graph = generate_mapper_graph(activations, projected_activations, multipass_cover, clusterer, refit_cover=False)
        # g = graph_to_dict_enhanced(g_multipass)
    else:

        mapper = km.KeplerMapper(verbose=verbose)
        # projected_activations = mapper.fit_transform(activations, projection='l2norm')

        graph = mapper.map(projected_activations, activations,
                        clusterer=clusterer,
                        cover=km.Cover(n_cubes=intervals, perc_overlap=overlap))

        iprint(f'Mapper graph generated with {len(graph["nodes"])} nodes and {len(graph["links"])} edges.')

    wirte_graph(graph, projected_activations, categorical_cols, intervals, overlap, eps, output_dir, output_fname, is_enhanced_cover=is_enhanced_cover)

    write_node_stats(graph, activations, np.array(categorical_cols['lable']), np.array(categorical_cols['predict']), intervals, overlap, eps, output_dir, output_fname, is_enhanced_cover=is_enhanced_cover)

    return graph

def get_node_id(node):
    interval_idx = node.interval_index
    cluster_idx = node.cluster_index
    node_id = "node"+str(interval_idx)+str(cluster_idx)
    return node_id

def wirte_graph(graph, filter_fn, categorical_cols, intervals, overlap, eps, output_dir, output_fname, is_enhanced_cover=False):
    g = {}
    g['nodes'] = {}
    g['edges'] = {}
    if is_enhanced_cover:
        for node in graph.nodes:
            node_id = get_node_id(node)
            g['nodes'][node_id] = [int(m) for m in list(node.members)]
        for k in graph.edges:
            node1_id, node2_id = get_node_id(k[0]), get_node_id(k[1])
            if node1_id not in g['edges']:
                g['edges'][node1_id] = []
            g['edges'][node1_id].append(node2_id)
    else: 
        for k in graph['nodes']:
            g['nodes'][k] = graph['nodes'][k]
        for k in graph['links']:
            g['edges'][k] = graph['links'][k]

    for node_id in g['nodes']:
        vertices = g['nodes'][node_id]
        node = {}
        node['categorical_cols_summary'] = {}
        node['vertices'] = vertices
        node['avgs'] = {}
        node['avgs']['lens'] = np.mean(filter_fn[vertices])
        for col in categorical_cols:
            data_categorical_i = categorical_cols[col].iloc[vertices]
            node['categorical_cols_summary'][col] = data_categorical_i.value_counts().to_dict()
        g['nodes'][node_id] = node
    g['categorical_cols'] = list(categorical_cols)
    numerical_col_keys = ['lens']
    g['numerical_col_keys'] = list(numerical_col_keys)

    if is_enhanced_cover:
        filename = 'mapper_' + str(output_fname) + '_' + str(intervals) + '_' + str(overlap) + '_' + '{:.2f}'.format(eps) + '_enhanced.json'
    else:
        filename = 'mapper_' + str(output_fname) + '_' + str(intervals) + '_' + str(overlap) + '_' + '{:.2f}'.format(eps) + '.json'

    with open(os.path.join(output_dir, filename), 'w') as fp:
        json.dump(g, fp)

def write_node_stats(graph, activations, ground_labels, pred_labels, intervals, overlap, eps, output_dir, output_fname, is_enhanced_cover=False):
    node_stats = defaultdict(NodeStat)

    if is_enhanced_cover:

        for node in graph.nodes:
            node_name = get_node_id(node)
            member_indices = [int(m) for m in list(node.members)]
            node_activations = activations[member_indices]
            node_ground_labels = ground_labels[member_indices]
            node_pred_labels = pred_labels[member_indices]

            node_l2norm = np.mean(np.linalg.norm(node_activations, axis=1))
            misclassification_rate = np.mean(node_ground_labels != node_pred_labels)
            class_compositions = np.bincount(node_ground_labels, minlength=10) / len(node_ground_labels)

            node_stats[node_name] = NodeStat(node_l2norm, misclassification_rate, list(class_compositions), len(member_indices))
            
    else:
        # compute node statistics
        for node_name, member_indices in graph['nodes'].items():
            node_activations = activations[member_indices]
            node_ground_labels = ground_labels[member_indices]
            node_pred_labels = pred_labels[member_indices]

            node_l2norm = np.mean(np.linalg.norm(node_activations, axis=1))
            misclassification_rate = np.mean(node_ground_labels != node_pred_labels)
            class_compositions = np.bincount(node_ground_labels, minlength=10) / len(node_ground_labels)

            node_stats[node_name] = NodeStat(node_l2norm, misclassification_rate, list(class_compositions), len(member_indices))

    node_stats_copy = {}
    for node_name in node_stats:
        node_stats_copy[node_name] = node_stats[node_name].__dict__

    if is_enhanced_cover:
        filename = 'node-stats_' + str(output_fname) + '_' + str(intervals) + '_' + str(overlap) + '_' + '{:.2f}'.format(eps) + '_enhanced.json' 
    else:
        filename = 'node-stats_' + str(output_fname) + '_' + str(intervals) + '_' + str(overlap) + '_' + '{:.2f}'.format(eps) + '.json'   

    with open(os.path.join(output_dir, filename), 'w') as fp:
        json.dump(node_stats_copy, fp)

if __name__ == '__main__':
    # # load the dataset (point cloud)
    activation_path = './data'
    filename = 'output_valset_mismatched'
    mapper_output_fname = filename
    mapper_output_dir = './graphData'
    activations_pd = pd.read_csv(os.path.join(activation_path, filename+'.csv'))
    # call the function to compute mapper graph
    intervals = 40
    overlap = 0.4
    eps = None
    activations = np.array(activations_pd.iloc[:, 3:])
    # 'lable', 'predict', 'correct'
    categorical_cols = {'lable':activations_pd['lable'], 'predict':activations_pd['predict'], 'correct':activations_pd['correct']}
    mapper_graph = create_mapper_graph(activations, intervals, overlap,categorical_cols, mapper_output_dir, mapper_output_fname, eps=eps, verbose=0, is_enhanced_cover=True)
    pass