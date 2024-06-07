import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])


data_set = np.genfromtxt("data_set.csv", delimiter = ",")

X = data_set[:, [0, 1]]

K = 4

def initialize_parameters(X, K):
    means = np.genfromtxt("initial_centroids.csv", delimiter=",")
    assignments = np.argmin(dt.cdist(X, means), axis = 1)
    covariances = np.array([np.cov(X[assignments == k], rowvar = False) for k in range(K)])
    priors = np.array([np.mean(assignments == k) for k in range(K)])
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

def em_clustering_algorithm(X, K, means, covariances, priors):
    for i in range(100):
        H = np.array([stats.multivariate_normal.pdf(X, mean = means[k], cov = covariances[k]) * priors[k] for k in range(K)]).T
        H /= np.sum(H, axis=1, keepdims=True)

        priors = np.sum(H, axis=0) / X.shape[0]
        means = np.array([np.sum(H[:, c, None] * X, axis=0) / np.sum(H, axis=0)[c] for c in range(K)])
        covariances = np.array([np.matmul((X - means[c]).T, H[:, c, None] * (X - means[c])) / np.sum(H, axis=0)[c] for c in range(K)])
        assignments = np.argmax(H, axis=1)

    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):

    cluster_colors = np.array(["#33a02c","#ff7f00", "#e31a1c", "#1f78b4"])

    x1_interval = np.linspace(-8, +8, 1601)
    x2_interval = np.linspace(-8, +8, 1601)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

    for c in range(K):
        original = stats.multivariate_normal.pdf(X_grid, mean = group_means[c], cov = group_covariances[c])
        original = original.reshape((len(x1_interval), len(x2_interval)))

        em = stats.multivariate_normal.pdf(X_grid, mean = means[c], cov = covariances[c])
        em = em.reshape((len(x1_interval), len(x2_interval)))

        plt.plot(X[assignments == c, 0], X[assignments == c, 1], ".", color = cluster_colors[c], markersize = 10)
        plt.contour(x1_grid, x2_grid, original, levels = [0.01], linestyles="dashed")
        plt.contour(x1_grid, x2_grid, em, levels = [0.01], colors = cluster_colors[c])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)
