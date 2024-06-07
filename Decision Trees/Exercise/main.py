import matplotlib.pyplot as plt
import numpy as np

data_set_train = np.genfromtxt("data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("data_set_test.csv", delimiter = ",", skip_header = 1)

X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

minimum_value = 1.5
maximum_value = 5.1
step_size = 0.001
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

def decision_tree_regression_train(X_train, y_train, P):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}

    node_indices[1] = np.array(range(len(y_train)))
    is_terminal[1] = False
    need_split[1] = True

    while True:
        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break

        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_means[split_node] = np.mean(y_train[data_indices])
            node_features[split_node] = 1
            if len(data_indices) <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(X_train[data_indices, 0]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2
                split_errors = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices, 0] > split_positions[s]]
                    right_indices = data_indices[X_train[data_indices, 0] <= split_positions[s]]
                    g_left = np.mean(y_train[left_indices])
                    g_right = np.mean(y_train[right_indices])
                    error = (1/len(data_indices)) * (np.sum((y_train[left_indices] - g_left)**2)
                                                     + np.sum((y_train[right_indices] - g_right)**2))
                    split_errors[s] = error

                node_splits[split_node] = split_positions[np.argmin(split_errors)]

                left_indices = data_indices[X_train[data_indices, 0] > node_splits[split_node]]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                right_indices = data_indices[X_train[data_indices, 0] <= node_splits[split_node]]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

    return(is_terminal, node_features, node_splits, node_means)

def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    y_hat = np.repeat(0.0, len(X_query))
    for i in range(len(X_query)):
        idx = 1
        while True:
            if is_terminal[idx]:
                y_hat[i] = node_means[idx]
                break
            else:
                if X_query[i] > node_splits[idx]:
                    idx = 2 * idx
                else:
                    idx = 2 * idx + 1
    return(y_hat)

def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    terminal_nodes = [key for key, values in is_terminal.items() if values == True]
    for terminal_node in terminal_nodes:
        idx = terminal_node
        rules = np.array([])
        while idx > 1:
            parent = np.floor(idx/2)
            if idx % 2 == 0:
                rules = np.append(rules, "x{:d} > {:.2f}".format(node_features[parent], node_splits[parent]))
            else:
                rules = np.append(rules, "x{:d} <= {:.2f}".format(node_features[parent], node_splits[parent]))
            idx = parent
        rules = np.flip(rules)
        print("Node {:02}: {} => {}".format(terminal_node, rules, node_means[terminal_node]))

P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)
