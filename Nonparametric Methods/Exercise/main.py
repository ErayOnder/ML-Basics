import math
import matplotlib.pyplot as plt
import numpy as np

data_set_train = np.genfromtxt("data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("data_set_test.csv", delimiter = ",", skip_header = 1)

x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start = minimum_value, stop = maximum_value, step = 0.001)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    g = np.asarray([np.mean(y_train[(left_borders[b] < x_train) & (x_train <= right_borders[b])]) for b in range(len(left_borders))])
    indices = np.digitize(x_query, bins=right_borders, right=True)
    y_hat = g[indices]
    return(y_hat)
    
bin_width = 0.35
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))

def running_mean_smoother(x_query, x_train, y_train, bin_width):
    y_hat = np.asarray([np.mean(y_train[(x - 0.5 * bin_width < x_train) & (x_train <= x + 0.5 * bin_width)]) for x in x_query])
    return(y_hat)

bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))

def kernel_smoother(x_query, x_train, y_train, bin_width):
    y_hat = np.asarray([np.sum(y_train * (1 / np.sqrt(2 * math.pi)) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2))
                        / np.sum((1 / np.sqrt(2 * math.pi)) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2))
                        for x in x_query])
    return(y_hat)

bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
