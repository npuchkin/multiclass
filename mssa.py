# coding: utf-8
#
# Nikita Puchkin
#
# Adaptive multiclass nearest neighbor classifier
#
#--------------------------------------------------------------------------

import numpy as np

#--------------------------------------------------------------------------
# Auxiliary functions
#--------------------------------------------------------------------------

def standard_k_agg(x):
    return 1 * (x < 1+1e-6)


def rectangular(x):
    return (np.abs(x) < 1 + 1e-6)


# KL-divergence between two Bernoulli distributions
# with parameters theta1 and theta2
def kl_div(theta1, theta2):
    trunc_theta1 = np.maximum(np.minimum(theta1, 1-1e-7), 1e-7)
    trunc_theta2 = np.maximum(np.minimum(theta2, 1-1e-7), 1e-7)
    
    return trunc_theta1 * np.log(trunc_theta1/trunc_theta2)+ (1 - trunc_theta1) * np.log((1 - trunc_theta1)/(1 - trunc_theta2))


def truncation(x, threshold):
    
    return np.minimum( np.maximum(threshold, x), 1 - threshold )


#--------------------------------------------------------------------------
# Multiclass spatial stagewise aggregation procedure
# of weighted nearest neighbor estimates
#--------------------------------------------------------------------------
#
# n_neighbors -- array of number of neighbors
# k_agg -- aggregation kernel
# loc_kernel -- localization kernel
#

class MSSA():
    
    def __init__(self, n_neighbors, agg_kernel=standard_k_agg, loc_kernel=rectangular, norm=2):
        # number of neighbors
        self.n_neighbors = n_neighbors
        # aggregation kernel
        self.agg_kernel = agg_kernel
        # localizing kernel
        self.loc_kernel = loc_kernel
        # norm
        self.norm = norm
    
    # X_train -- (n_samples x n_features)-array with design points
    # y_train -- (n_samples)-array of labels
    # x_test -- (n_features)-array of features of a test point
    def compute_weighted_knn(self, X_train, y_train, x_test):
        
        # number of training points
        n = X_train.shape[0]
        # number of k-NN estimates
        K = self.n_neighbors.shape[0]
        # euclidean distances
        distances = np.linalg.norm(X_train - np.outer(np.ones(n), x_test), ord=self.norm, axis=1)
        # indices of points from the closest point to the most distant one 
        closest = np.argsort(distances)
        # bandwidths
        h = distances[closest[self.n_neighbors]]
        # localizing weights for all k-NN estimates
        weights = self.loc_kernel(np.divide(np.outer(distances, np.ones(K)), np.outer(np.ones(n), h)))
        # weighted k_NN estimates
        N = np.sum(weights, axis=0)
        S = np.sum(np.multiply(weights, np.outer(y_train, np.ones(K))), axis=0)
        knn_estimates = np.divide(S, N)
        
        return knn_estimates
        
    def tune_critical_values(self, X, x_test, n_classes, n_samples=1000, confidence=0.95):
        
        alpha = 1 - confidence
        K = self.n_neighbors.shape[0]
        kl = np.zeros((1, K))
    
        for i in range(n_samples):
            # generate random labels
            y_random = np.random.randint(2, size=X.shape[0])
            
            # compute k-NN estimates
            knn_estimates = self.compute_weighted_knn(X, y_random, x_test)
            knn_estimates = truncation(knn_estimates, 0.5 / n_classes)
            knn_shifted = np.append(np.zeros(1), knn_estimates[:-1], axis=0)
            
            # compute KL-divergence between sequential estimates
            kl = np.append(kl, kl_div(knn_estimates, knn_shifted).reshape(1, -1), axis=0)
            
        
        # empirical alpha-quantile
        kl_sorted = np.sort(kl, axis=0)
        q = int(n_samples * alpha / K) + 1

        kl_quantile = kl_sorted[-q, :]
        
        return np.maximum(kl_quantile.reshape(-1), 1e-7)
        
    def aggregate(self, knn_estimates, critical_values):
        
        K = knn_estimates.shape[0]
        ssa_estimate = knn_estimates[0]
        
        for k in range(1, K):
            gamma = self.agg_kernel(kl_div(ssa_estimate, knn_estimates[k]) / critical_values[k])
            ssa_estimate = gamma * knn_estimates[k] + (1 - gamma) * ssa_estimate
        
        return ssa_estimate
        
    def predict(self, X_train, y_train, x_test, critical_values):
        
        n_classes, classes = self.get_classes(y_train)
        
        ssa_estimates = np.zeros(n_classes)
        knn_estimates = np.empty((self.n_neighbors.shape[0], 1))
        
        # OvA
        for j in range(n_classes):

            # transform labels to binary
            y_bin = 1 * (y_train == classes[j])

            # computation of kNN estimates
            knn_estimates_j = self.compute_weighted_knn(X_train, y_bin, x_test)
            knn_estimates_j = truncation(knn_estimates_j, 0.5 / n_classes)
            # computation of SSA estimate
            ssa_estimates[j] = self.aggregate(knn_estimates_j, critical_values)
            
            knn_estimates = np.append(knn_estimates, knn_estimates_j.reshape(-1, 1), axis=1)
        
        # return MSSA prediction and k-NN predicitons
        return classes[np.argmax(ssa_estimates)], classes[np.argmax(knn_estimates[:, 1:], axis=1)]



    # Returns the number of classes and the classes itself
    def get_classes(self, y):
        
        # Define the number of classes
        classes = np.unique(y)
        n_classes = classes.shape[0]
        
        return n_classes, classes    