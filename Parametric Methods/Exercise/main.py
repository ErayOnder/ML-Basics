import numpy as np
import pandas as pd

X = np.genfromtxt("data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("class_labels.csv", delimiter = ",", dtype = int)

def train_test_split(X, y):
    X_train = X[:50000]
    y_train = y[:50000]
    X_test = X[50000:]
    y_test = y[50000:]
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def estimate_prior_probabilities(y):
    K = np.max(y)
    class_priors = [np.mean(y == c+1) for c in range(K)]
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)

def estimate_nucleotide_probabilities(X, y):
    K = np.max(y)
    D = np.size(X[0])

    pAcd = np.zeros((K, D))
    pCcd = np.zeros((K, D))
    pGcd = np.zeros((K, D))
    pTcd = np.zeros((K, D))

    for c in range(K):
        X_new = X[y==c+1]

        pAcd[c] = np.mean(X_new == 'A', axis=0)
        pCcd[c] = np.mean(X_new == 'C', axis=0)
        pGcd[c] = np.mean(X_new == 'G', axis=0)
        pTcd[c] = np.mean(X_new == 'T', axis=0)

    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)

def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    N = X.shape[0]
    K = np.size(class_priors)
    score_values = np.zeros((N,K))
    map = {'A':pAcd,'C':pCcd,'G':pGcd,'T':pTcd}

    for n in range(N):
        product1 = 1
        product2 = 1
        for d in range(np.size(pAcd[0])):
            ch = X[n][d]
            product1 *= map[ch][0][d]
            product2 *= map[ch][1][d]
        score_values[n][0] = np.log(product1 * class_priors[0])
        score_values[n][1] = np.log(product2 * class_priors[1])

    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)

def calculate_confusion_matrix(y_truth, scores):
    y_pred = np.where(scores[:,0] > scores[:,1], 1, 2)
    confusion_matrix = pd.crosstab(y_pred.T, y_truth.T).values
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
