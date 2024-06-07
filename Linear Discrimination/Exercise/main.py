import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = np.genfromtxt("data_points.csv", delimiter = ",") / 255
y = np.genfromtxt("class_labels.csv", delimiter = ",").astype(int)

i1 = np.hstack((np.reshape(X[np.where(y == 1)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 2)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 3)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 4)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 5)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 6)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 7)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 8)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 9)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 10)[0][0:5], :], (28 * 5, 28))))

fig = plt.figure(figsize = (10, 5))
plt.axis("off")
plt.imshow(i1, cmap = "gray")
plt.show()
fig.savefig("images.pdf", bbox_inches = "tight")

def train_test_split(X, y):
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[60000:]
    y_test = y[60000:]
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def sigmoid(X, W, w0):
    scores = 1 / (1 + np.exp(-(np.matmul(X, W) + w0)))
    return(scores)

def one_hot_encoding(y):
    Y = np.zeros((y.shape[0], np.max(y)), dtype = int)
    Y[np.arange(y.shape[0]),y-1] = 1
    return(Y)

np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.001, high = 0.001, size = (D, K))
w0_initial = np.random.uniform(low = -0.001, high = 0.001, size = (1, K))

def gradient_W(X, Y_truth, Y_predicted):
    gradient = - np.matmul(X.T, (Y_truth-Y_predicted)*Y_predicted*(1-Y_predicted))
    return(gradient)

def gradient_w0(Y_truth, Y_predicted):
    gradient = np.sum(-(Y_truth-Y_predicted)*Y_predicted*(1-Y_predicted), axis=0)
    return(gradient)

def discrimination_by_regression(X_train, Y_train, W_initial, w0_initial):
    eta = 0.15 / X_train.shape[0]
    iteration_count = 250

    W = W_initial
    w0 = w0_initial
    objective_values = []
    for i in range(iteration_count):
        y_predicted = sigmoid(X_train, W, w0)
        grad_w = gradient_W(X_train, Y_train, y_predicted)
        grad_w0 = gradient_w0(Y_train, y_predicted)
        W -= eta * grad_w
        w0 -= eta * grad_w0

        objective_values = np.append(objective_values, 0.5 * np.sum((Y_train - y_predicted)**2))
    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train,
                                                       W_initial, w0_initial)
print(W)
print(w0)
print(objective_values[0:10])

fig = plt.figure(figsize = (10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
fig.savefig("iterations.pdf", bbox_inches = "tight")

def calculate_predicted_class_labels(X, W, w0):
    y_exp = np.exp((np.matmul(X, W) + w0))
    y_softmax = y_exp / np.sum(y_exp, axis=1)[:, None]
    y_predicted = np.argmax(y_softmax, axis=1) + 1
    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)
print(y_hat_train)

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)
print(y_hat_test)


def calculate_confusion_matrix(y_truth, y_predicted):
    confusion_matrix = pd.crosstab(y_predicted.T, y_truth.T, rownames = ["Y Predicted"], colnames = ["Y Truth"]).to_numpy()
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)
