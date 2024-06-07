import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities = np.genfromtxt("predicted_probabilities.csv", delimiter = ",")

def calculate_threholds(predicted_probabilities):
    sorted_prob = np.sort(predicted_probabilities)
    thresholds = np.zeros(len(predicted_probabilities) + 1)
    thresholds[0] = (sorted_prob[0] + 0)/2
    thresholds[-1] = (sorted_prob[-1] + 1)/2
    for i in range(1, len(sorted_prob)):
        thresholds[i] = (sorted_prob[i-1] + sorted_prob[i])/2
    return thresholds

thresholds = calculate_threholds(predicted_probabilities)
print(thresholds)

def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    fp_rates = np.zeros(len(thresholds))
    tp_rates = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        predicted_values = (predicted_probabilities >= threshold)
        fp_rates[i] = np.sum((predicted_values==1)&(true_labels==-1)) / np.sum(true_labels==-1)
        tp_rates[i] = np.sum((predicted_values==1)&(true_labels==1)) / np.sum(true_labels==1)
    return fp_rates, tp_rates

fp_rates, tp_rates = calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds)
print(fp_rates)
print(tp_rates)

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates, tp_rates)
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.show()
fig.savefig("roc_curve.pdf", bbox_inches = "tight")

def calculate_auroc(fp_rates, tp_rates):
    auroc = np.trapz(np.sort(np.sort(tp_rates)),np.sort(fp_rates))
    return auroc

auroc = calculate_auroc(fp_rates, tp_rates)
print("The area under the ROC curve is {}.".format(auroc))
