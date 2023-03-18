import torch
import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from scoring import *

class WordBank:
    def __init__(self, num_clusters=5):
        self.clusters = []
        self.GloVe = {}
        self.word_bank = []
        self.wb_embeddings = None
        self.n_clusters = num_clusters

    def load_word_vecs(self):
        # Load word vectors
        self.GloVe = {}
        with open("glove.6B/glove.6B.100d.txt", "r", encoding="utf-8") as vector_file:
            for line in vector_file:
                line_content = line.split()
                word = line_content[0]
                # There's probably a better way to read strings into a FloatTensor
                word_vec = torch.from_numpy(np.asarray(line_content[1:], "float32"))
                self.GloVe[word] = word_vec

    def load_word_bank(self):
        self.word_bank = []
        with open('bad_words.csv') as bad_words_csv:
            self.word_bank = list(csv.reader(bad_words_csv, delimiter=","))[0]
        #print(self.word_bank)

        # Create Word Embeddings Matrix
        final_word_bank = []
        for word in self.word_bank:
            word = word.lower()
            if word in self.GloVe:
                final_word_bank.append(word)


        self.wb_embeddings = torch.zeros((len(final_word_bank), 100))
        for i, word in enumerate(final_word_bank):
            if word.lower() in self.GloVe:
                self.wb_embeddings[i] = self.GloVe[word.lower()]
                
        self.word_bank = final_word_bank
        print(self.wb_embeddings.size())


    def create_clusters(self):
        # use k-means to auto-cluster the word bank

        word_vectors = self.wb_embeddings
        #print(word_vectors.shape)
        num_clusters = self.n_clusters

        clusterer = KMeans(n_clusters=num_clusters)
        clusterer.fit(word_vectors)

        self.clusters = clusterer.labels_
        #print(self.clusters)


    def calculate_score(self, embeddings, score_mode):
        """
        if DIST == 'dotp':
            dist_score = [dotp_similarity_score(embed) for embed in top_embeddings]
        elif DIST == 'dot':
            dist_score = [dot_similarity_score(embed) for embed in top_embeddings]
        elif DIST == 'distp':
            dist_score = [distancep_score(embed) for embed in top_embeddings]
        """
        if score_mode == 'dist':
            dist_score = [distance_score(embed, 
                                        self.wb_embeddings, 
                                        self.clusters, 
                                        self.n_clusters) for embed in embeddings]
        else:
            print('DIST error')
            return None
        return dist_score
    
    def visualize_wbank(self):
        color_pool = ['blue', 'red', 'green', 'pink', 'orange', 'black', 'yellow']
        # these labels are colors
        colors = []
        for label in self.clusters:
            colors.append(color_pool[label])

        low_dim_wb = TSNE(n_components=2, 
                        learning_rate='auto',
                        init='pca', 
                        perplexity=3).fit_transform(self.wb_embeddings)

        plt.scatter(low_dim_wb[:,0], low_dim_wb[:,1], color=colors)

        for c, label, x, y in zip(colors, self.word_bank[:], low_dim_wb[:, 0], low_dim_wb[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=6, color=c, fontweight='bold')
        plt.show()
