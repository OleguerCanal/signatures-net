import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from signaturesnet.utilities.io import csv_to_tensor, read_model

def stylize_axes(ax, title, xlabel, ylabel):
    """Customize axes spines, title, labels, ticks, and ticklabels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def accuracy(prediction, label):
    assert(prediction.shape == label.shape)
    assert(prediction.dtype == torch.int64)
    assert(label.dtype == torch.int64)
    return (torch.sum(prediction == label).float()/torch.numel(prediction))*100.

def false_realistic(prediction, label):
    assert(prediction.shape == label.shape)
    assert(prediction.dtype == torch.int64)
    # assert(label.dtype == torch.int64)
    # print(label[prediction == 1])
    # print(torch.sum(label[prediction == 1] == 0))
    return torch.true_divide(torch.sum(label[prediction == 1] == 0), (torch.numel(label)))*100 

def false_random(prediction, label):
    assert(prediction.shape == label.shape)
    assert(prediction.dtype == torch.int64)
    # assert(label.dtype == torch.int64)
    return torch.true_divide(torch.sum(label[prediction == 0] == 1), (torch.numel(label)))*100 

def plot_metric_vs_mutations_classifier(guess, label, num_muts_list, plot_path = None, show=False):
    fig, axs = plt.subplots(3, figsize=(8,4))
    fig.suptitle("Detector Performance")
    num_muts = np.unique(num_muts_list.detach().numpy())[:-1]
    marker_size = 3
    line_width = 0.5
    values = np.zeros((3, len(num_muts)))
    for i, num_mut in enumerate(num_muts):
        indexes = num_muts_list == num_mut
        values[0,i] = accuracy(label=label[indexes], prediction=guess[indexes])
        values[1,i] = false_realistic(label=label[indexes], prediction=guess[indexes])
        values[2,i] = false_random(label=label[indexes], prediction=guess[indexes])
    axs[0].plot(np.log10(num_muts), values[0,:], marker='o',linewidth=line_width, markersize=marker_size)
    axs[1].plot(np.log10(num_muts), values[1,:], marker='o',linewidth=line_width, markersize=marker_size)
    axs[2].plot(np.log10(num_muts), values[2,:], marker='o',linewidth=line_width, markersize=marker_size)
    y_labels = ["Accuracy (%)", "False Realistic (%)", "False Random (%)"]
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', 'log(N)', y_labels[i])
    # stylize_axes(axs, '', 'log(N)', y_labels)
    fig.tight_layout()
    if show:
        plt.show()
    if plot_path is not None:
        fig.savefig(plot_path)

def get_dist_vec(reals, reconstructions):
    N = len(reals)
    D = torch.zeros(N)
    for i in range(N):
        D[i] = nn.MSELoss()(reals[i], reconstructions[i])
    return D

# Classifier performance
num_mut = csv_to_tensor("../data/datasets/detector/test_num_mut.csv", device='cpu')
inputs = csv_to_tensor("../data/datasets/detector/test_input.csv", device='cpu')
label_classifier = csv_to_tensor("../data/datasets/detector/test_label.csv", device='cpu').to(torch.int64)

classifier = read_model("../trained_models/vae_classifier/ae_best_3_AUC")
classifier_guess, mu, var = classifier(inputs, num_mut)

distances = get_dist_vec(inputs, classifier_guess)
print(distances)
distance_cutoff = 0.000005

classification_results = (distances <= distance_cutoff).to(torch.int64).reshape(-1,1)
print(classification_results)

plot_metric_vs_mutations_classifier(classification_results, label_classifier, num_mut, plot_path = 'test_ae_classifier/detector_performance_ae_best_3_AUC_5e6.png')
