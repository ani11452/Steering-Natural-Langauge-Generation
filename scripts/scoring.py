import torch
import numpy as np
import csv
import random

eps = 0.00000000000001
NUM_SAMPLES = 50000

# Global variables cluster_list, p_look_up made in set_up()

class Node(object):
    def __init__(self, d):
        self.data = d
        self.left = None
        self.right = None

def sortedArrayToBST(arr):
    if not arr:
        return None
    
    mid = (len(arr)) // 2
    root = Node(arr[mid])
    root.left = sortedArrayToBST(arr[:mid])
    root.right = sortedArrayToBST(arr[mid+1:])
    return root

def create_p(samples):
    list_ascending = sorted(samples.tolist())
    list_descending = sorted(samples.tolist(), reverse=True)
    p_dict = {val: float(i/len(samples)) for i, val in enumerate(list_ascending)}
    p_dict[float("-inf")] = 0.0
    p_dict[float('inf')] = 1.0
    bst = sortedArrayToBST(list_descending)
    return bst, p_dict

def get_p_value(bst, value, p_dict):
    ran = [float('inf'), float('-inf')]
    while True:
        if value < bst.data:
            ran[0] = min(ran[0], bst.data)
            if not bst.right:
                return p_dict[ran[0]]
            bst = bst.right
        elif value >= bst.data:
            ran[1] = max(ran[1], bst.data)
            if not bst.left:
                return p_dict[ran[0]]
            bst = bst.left

def set_up(wb_embeddings, clusters, num_clusters, GloVe, word_bank):
    global cluster_list
    cluster_list = [[] for i in range(num_clusters)]
    for i, cluster in enumerate(clusters):
        cluster_list[cluster].append(i)

    vocab = list(GloVe.values())
    idx_to_sample = torch.zeros((NUM_SAMPLES, len(word_bank)))

    sample = 0
    while sample < NUM_SAMPLES:
        word = random.choice(vocab)
        distances = wb_embeddings - word
        distances = torch.linalg.norm(distances, dim=1)
        idx_to_sample[sample] = distances
        sample += 1
    idx_to_sample = idx_to_sample.reshape(len(word_bank), NUM_SAMPLES)
    
    global p_look_up
    p_look_up = []
    for i in range(len(word_bank)):
        p_look_up.append(create_p(idx_to_sample[i])) 

def distance_score(embedding, wb_embeddings, clusters, num_clusters):
    #print('correctly using distance score')
    distances = wb_embeddings - embedding
    distances = torch.linalg.norm(distances, dim=1)
    
    cluster_means = [0] * num_clusters
    cluster_counts = [0] * num_clusters
    for cluster, dist in zip(clusters, distances):
        cluster_means[cluster] += dist
        cluster_counts[cluster] += 1
    
    cluster_means = [cluster_means[idx]/cluster_counts[idx] for idx in range(len(cluster_means))]
    # print(cluster_means)
    # print("Using distance")
    return min(cluster_means)

def dot_score(word_emb, wb_embeddings, clusters, num_clusters):
    similarities = torch.matmul(wb_embeddings, word_emb)

    cluster_means = [0] * num_clusters
    cluster_counts = [0] * num_clusters
    for cluster, dist in zip(clusters, similarities):
        cluster_means[cluster] += dist
        cluster_counts[cluster] += 1

    cluster_means = [cluster_means[idx]/cluster_counts[idx] for idx in range(len(cluster_means))]
    # print("using dot product")
    return 1 / (max(cluster_means) + eps)
    
def cluster_stat(embedding, distances, cluster):
    avg = 0
    for word_idx in cluster:
        bst, p_dict = p_look_up[word_idx]
        distance = distances[word_idx]
        p = get_p_value(bst, distance, p_dict)
        avg += p
    return float(avg / len(cluster))

def statistics(word_emb, wb_embeddings, clusters, num_clusters):
    scores = []
    distances = wb_embeddings - word_emb
    distances = torch.linalg.norm(distances, dim=1).tolist()
    for cluster in cluster_list:
        scores.append(cluster_stat(word_emb, distances, cluster))
    return min(scores)

def dot_similarity(wb_embeddings, word_emb):
    similarities = torch.matmul(wb_embeddings, word_emb)
    return similarities

"""
Implement dotp, distp, BST, & other scoring functions below
"""
