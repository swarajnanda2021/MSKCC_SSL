

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy import optimize
from sklearn.manifold import SpectralEmbedding

class UMAPRepresentation(object):
    def __init__(self, data_path, filename, n_neighbors=15, min_dist=0.5, n_components=2, learning_rate=1, max_iter=200):
        self.data_path = data_path
        self.filename = filename
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.load_data()
        self.calculate_distances()
        self.initialize_spectral_embedding()
        self.calculate_probabilities()

    def load_data(self):
        expr = pd.read_csv(self.data_path + self.filename, sep='\t')
        self.X_train = expr.values[:, 0:expr.shape[1]-1]
        self.Y_train = expr.values[:, expr.shape[1]-1]
        self.X_train = np.log(self.X_train + 1)

    def calculate_distances(self):
        self.distances = np.square(euclidean_distances(self.X_train, self.X_train))
        self.rho = [sorted(self.distances[i])[1] for i in range(self.distances.shape[0])]

    def initialize_spectral_embedding(self):
        model = SpectralEmbedding(n_components=2, n_neighbors=50)
        self.specembed = model.fit_transform(np.log(1 + self.X_train))

    def plot_initial_embedding(self):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.specembed[:, 0], self.specembed[:, 1], c=self.Y_train.astype(int), cmap='tab10', s=50)
        plt.title('Laplacian Eigenmap', fontsize=14)
        plt.xlabel("LAP1", fontsize=12)
        plt.ylabel("LAP2", fontsize=12)
        plt.show()

    def prob_high_dim_umap(self, sigma, dist_row):
        d = self.distances[dist_row] - self.rho[dist_row]
        d[d < 0] = 0
        return np.exp(-d / sigma)
    
    def func(self, x, min_dist):
        y = []
        for i in range(len(x)):
            if x[i] <= min_dist:
                y.append(1)
            else:
                y.append(np.exp(-x[i] + min_dist))
        return y

    def k(self, prob):
        return np.power(2, np.sum(prob))

    def sigma_binary_search(self, k_of_sigma, fixed_k):
        sigma_low, sigma_high = 0, 1000
        for i in range(20):
            sigma_mid = (sigma_low + sigma_high) / 2
            if k_of_sigma(sigma_mid) < fixed_k:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            if np.abs(fixed_k - k_of_sigma(sigma_mid)) <= 1e-5:
                break
        return sigma_mid

    def calculate_probabilities(self):
        n = self.X_train.shape[0]
        self.prob = np.zeros((n, n))
        self.sigma_array = []
        for dist_row in range(n):
            func = lambda sigma: self.k(self.prob_high_dim_umap(sigma, dist_row))
            bin_search_result = self.sigma_binary_search(func, self.n_neighbors)
            self.prob[dist_row] = self.prob_high_dim_umap(bin_search_result, dist_row)
            self.sigma_array.append(bin_search_result)
        self.P = (self.prob + np.transpose(self.prob)) / 2 # This is symmetric prob, but it should not be this one. But this one has been used in the tutorial I consulted.

    def dist_low_dim(self, x, a, b):
        return 1 / (1 + a * x ** (2 * b))

    def prob_low_dim_umap(self, Y):
        euclid_distances = euclidean_distances(Y, Y)
        Q = (1 + self.a * euclid_distances ** (2 * self.b))
        return np.power(Q, -1)

    def CE(self, P, Y):
        Q = self.prob_low_dim_umap(Y)
        CE_term1 = -P * np.log(Q + 0.01)
        CE_term2 = - (1 - P) * np.log(1 - Q + 0.01)
        return CE_term1 + CE_term2

    def CE_gradient(self, P, Y):
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        inv_dist = np.power((1 + self.a * euclidean_distances(Y, Y) ** (2 * self.b)), -1)
        Q = np.dot((1 - P), np.power((1 + self.a * euclidean_distances(Y, Y) ** (2 * self.b)), -1))
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q, axis=1, keepdims=True)
        fact = np.expand_dims(self.a * P * (1e-8 + np.square(euclidean_distances(Y, Y))) ** (self.b - 1) - Q, 2)
        return 2 * self.b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis=1)

    def fit(self):
        x = np.linspace(0, 3, 100)
        p, _ = optimize.curve_fit(self.dist_low_dim, x, self.func(x, self.min_dist))
        self.a, self.b = p[0], p[1]
        np.random.seed(12345)
        #Y = np.random.normal(loc=0, scale=1, size=(self.X_train.shape[0], self.n_components)) # random initialization
        model = SpectralEmbedding(n_components = N_LOW_DIMS, n_neighbors = 50) # Laplacian initialization
        Y = model.fit_transform(np.log(self.X_train + 1))
        self.CE_array = []
        for i in range(self.max_iter):
            Y -= self.learning_rate * self.CE_gradient(self.P, Y)
            CE_current = np.sum(self.CE(self.P, Y)) / 1e+5
            self.CE_array.append(CE_current)
        return Y


################################################################################################################################################################


class TSNERepresentation(object):
    def __init__(self, data_path, filename, perplexity=30, n_components=2, learning_rate=0.6, max_iter=400):
        self.data_path = data_path
        self.filename = filename
        self.perplexity = perplexity
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.load_data()
        self.calculate_distances()
        self.calculate_probabilities()

    def load_data(self):
        expr = pd.read_csv(self.data_path + self.filename, sep='\t')
        self.X_train = expr.values[:, 0:expr.shape[1]-1]
        self.Y_train = expr.values[:, expr.shape[1]-1]
        self.X_train = np.log(self.X_train + 1)

    def calculate_distances(self):
        self.distances = np.square(euclidean_distances(self.X_train, self.X_train))

    def prob_high_dim(self, sigma, dist_row):
        exp_dist = np.exp(-self.distances[dist_row] / (2 * sigma**2))
        exp_dist[dist_row] = 0
        unsymm_prob = exp_dist / np.sum(exp_dist)
        return unsymm_prob

    def calculate_perplexity(self, prob):
        return np.power(2, -np.sum([p * np.log2(p) for p in prob if p != 0]))

    def sigma_binary_search(self, perp_of_sigma):
        sigma_low, sigma_high = 0, 1000
        for i in range(20):
            sigma_mid = (sigma_low + sigma_high) / 2
            if perp_of_sigma(sigma_mid) < self.perplexity:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            if np.abs(self.perplexity - perp_of_sigma(sigma_mid)) <= 1e-5:
                break
        return sigma_mid

    def calculate_probabilities(self):
        n = self.X_train.shape[0]
        self.prob = np.zeros((n, n))
        self.sigma_array = []
        for dist_row in range(n):
            func = lambda sigma: self.calculate_perplexity(self.prob_high_dim(sigma, dist_row))
            bin_search_result = self.sigma_binary_search(func)
            self.prob[dist_row] = self.prob_high_dim(bin_search_result, dist_row)
            self.sigma_array.append(bin_search_result)
        self.P = self.prob + self.prob.T

    def prob_low_dim(self, Y):
        euclid_distances = euclidean_distances(Y, Y)
        q_ij = np.power(1 + np.power(euclid_distances, 2), -1)
        np.fill_diagonal(q_ij, 0)
        q_ij = q_ij / np.sum(q_ij, axis=1, keepdims=True)
        return q_ij

    def KL_divergence(self, Y):
        Q = self.prob_low_dim(Y)
        return self.P * np.log(self.P + 0.01) - self.P * np.log(Q + 0.01)

    def KL_divergence_gradient(self, Y):
        Q = self.prob_low_dim(Y)
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        inv_dist = np.power(1 + np.power(euclidean_distances(Y, Y), 2), -1)
        grad_formula = 4 * np.sum(np.expand_dims((self.P - Q), 2) * y_diff * np.expand_dims(inv_dist, 2), axis=1)
        return grad_formula

    def fit(self):
        np.random.seed(12345)
        Y = np.random.normal(loc=0, scale=1, size=(self.X_train.shape[0], self.n_components))
        KL_array = []
        for i in range(self.max_iter):
            Y -= self.learning_rate * self.KL_divergence_gradient(Y)
            KL_array.append(np.sum(self.KL_divergence(Y)))
        return Y, KL_array

    def plot_tsne(self, Y):
        plt.figure(figsize=(4, 4))
        plt.scatter(Y[:, 0], Y[:, 1], c=self.Y_train.astype(int), cmap='tab10', s=50)
        plt.title("tSNE on Cancer Associated Fibroblasts (CAFs): Programmed from Scratch", fontsize=14)
        plt.xlabel("tSNE1", fontsize=14)
        plt.ylabel("tSNE2", fontsize=14)
        plt.show()

    def plot_kl_divergence(self, KL_array):
        plt.figure(figsize=(4, 4))
        plt.plot(KL_array)
        plt.title("KL-divergence", fontsize=14)
        plt.xlabel("ITERATION", fontsize=14)
        plt.ylabel("KL-DIVERGENCE", fontsize=14)
        plt.show()



################################################################################################################################################################
if __name__ == '__main__': # Make sure you have the file 'bartoschek_filtered_expr_rpkm.txt' from 'https://github.com/NikolayOskolkov/HowUMAPWorks/blob/master/HowUMAPWorks.ipynb'
  # UMAP Usage example
  umap_rep = UMAPRepresentation('./UMAP/', 'bartoschek_filtered_expr_rpkm.txt')
  umap_rep.plot_initial_embedding()
  Y = umap_rep.fit()
  
  # Plot final embedding
  plt.scatter(Y[:, 0], Y[:, 1], c=umap_rep.Y_train.astype(int), cmap='tab10', s=50)
  plt.title('UMAP Representation', fontsize=14)
  plt.xlabel("UMAP1", fontsize=12)
  plt.ylabel("UMAP2", fontsize=12)
  plt.show()
  
  
  ################################################################################################################################################################
  
  # tSNE Usage example
  tsne_rep = TSNERepresentation('./UMAP/', 'bartoschek_filtered_expr_rpkm.txt')
  Y, KL_array = tsne_rep.fit()
  tsne_rep.plot_tsne(Y)
  tsne_rep.plot_kl_divergence(KL_array)
  
  
