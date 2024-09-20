import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc

from signaturesnet.utilities.io import csv_to_tensor, read_model, read_data_classifier

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

def plot_hist(subplot, fake_d_np, real_d_np, n):
    subplot.hist(fake_d_np, bins=np.linspace(0,10,100), alpha=0.5, label='Fake Distances', color='blue')
    subplot.hist(real_d_np, bins=np.linspace(0,10,100), alpha=0.5, label='Real Distances', color='orange')
    subplot.set_title('Histogram of Fake and Real Distances (n=%s)'%str(n))
    subplot.set_xlabel('Distance')
    subplot.set_ylabel('Frequency')
    # subplot.set_xlim([0.0, 10.0])
    subplot.legend()

def plot_roc(subplot, binary_labels, probabilities, n):
    # Compute ROC curve and ROC area
    print(binary_labels)
    fpr, tpr, thresholds = roc_curve(binary_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    subplot.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    subplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    subplot.set_xlim([0.0, 1.0])
    subplot.set_ylim([0.0, 1.05])
    subplot.set_xlabel('False Positive Rate')
    subplot.set_ylabel('True Positive Rate')
    subplot.set_title('Receiver Operating Characteristic (ROC) Curve (n=%s)'%str(n))
    subplot.legend(loc='lower right')

def plot_roc_and_hist_nummut(inps, guess, label, num_muts_list, plot_path = None, show=False):
    num_muts = np.unique(num_muts_list.detach().numpy())[:-1]
    # fig, axs = plt.subplots(len(num_muts),2, figsize=(12,6))
    # for i, num_mut in enumerate(num_muts):
    # First ROC
    fig, subplots = plt.subplots(nrows=len(num_muts), ncols=1, figsize=(6,15))
    for n, subplot in enumerate(subplots.flatten()):
        num_mut = num_muts[n]
        indexes = num_muts_list == num_mut
        inputs = inps[indexes.reshape(-1)]
        pred = guess[indexes.reshape(-1)]
        labels = label[indexes.reshape(-1)]
        distances = torch.clip(get_dist_vec(inputs, pred).to('cpu'), 0, 10.0)
        fake_d = distances[(labels == 0).squeeze(-1)]
        real_d = distances[(labels == 1).squeeze(-1)]
        probs = 1 - (distances / 10.0)  # very improvable
        plot_roc(subplot,labels.cpu().numpy(), probs.cpu().numpy(), num_mut)
    fig.tight_layout()
    if show:
        plt.show()
    if plot_path is not None:
        fig.savefig(plot_path + '_ROC.png')
    plt.close()

    # Second hist distances
    fig, subplots = plt.subplots(nrows=len(num_muts), ncols=1, figsize=(6,15))
    for n, subplot in enumerate(subplots.flatten()):
        num_mut = num_muts[n]
        indexes = num_muts_list == num_mut
        inputs = inps[indexes.reshape(-1)]
        pred = guess[indexes.reshape(-1)]
        labels = label[indexes.reshape(-1)]
        distances = torch.clip(get_dist_vec(inputs, pred).to('cpu'), 0, 10.0)
        fake_d = distances[(labels == 0).squeeze(-1)]
        real_d = distances[(labels == 1).squeeze(-1)]
        probs = 1 - (distances / 10.0)  # very improvable
        plot_hist(subplot,fake_d.cpu().numpy(), real_d.cpu().numpy(), num_mut)
    fig.tight_layout()
    if show:
        plt.show()
    if plot_path is not None:
        fig.savefig(plot_path + '_hist.png')
    plt.close()


    # values = np.zeros((3, len(num_muts)))
    # for i, num_mut in enumerate(num_muts):
    #     indexes = num_muts_list == num_mut
    #     values[0,i] = accuracy(label=label[indexes], prediction=guess[indexes])
    #     values[1,i] = false_realistic(label=label[indexes], prediction=guess[indexes])
    #     values[2,i] = false_random(label=label[indexes], prediction=guess[indexes])
    # axs[0].plot(np.log10(num_muts), values[0,:], marker='o',linewidth=line_width, markersize=marker_size)
    # axs[1].plot(np.log10(num_muts), values[1,:], marker='o',linewidth=line_width, markersize=marker_size)
    # axs[2].plot(np.log10(num_muts), values[2,:], marker='o',linewidth=line_width, markersize=marker_size)
    # y_labels = ["Accuracy (%)", "False Realistic (%)", "False Random (%)"]
    # for i, axes in enumerate(axs.flat):
    #     stylize_axes(axes, '', 'log(N)', y_labels[i])
    # stylize_axes(axs, '', 'log(N)', y_labels)
    

def get_dist_vec(reals, reconstructions):
    N = len(reals)
    D = torch.zeros(N)
    for i in range(N):
        D[i] = nn.MSELoss()(reals[i], reconstructions[i])
    return D

class DistanceBasedClassifier(nn.Module):
    def __init__(self, train_data):
        super(DistanceBasedClassifier, self).__init__()
        self.train_data = train_data
        
    @torch.no_grad()
    def forward(self, mutation_dist, num_mut=None):
        """nummut is not used in this model, but it is included for compatibility with other models
        """
        distances = torch.cdist(mutation_dist, self.train_data)
        # IDEA: Improvable doing -mean(top-k of -dist) to avoid outliers
        min_dist = torch.min(distances, dim=1).values * 10
        return min_dist

# Classifier performance
num_mut = csv_to_tensor("../data/datasets/detector/test_num_mut.csv", device='cpu')
inputs = csv_to_tensor("../data/datasets/detector/test_input.csv", device='cpu')
label_classifier = csv_to_tensor("../data/datasets/detector/test_label.csv", device='cpu').to(torch.int64)

# classifier = read_model("../trained_models/vae_classifier/ae_best_3_AUC")
# classifier_guess, mu, var = classifier(inputs, num_mut)

train_data, val_data = read_data_classifier(
    device='cpu',
    experiment_id="datasets/detector",
)

train_input = train_data.inputs
train_input = train_input[(train_data.labels == 1).squeeze(-1)]
train_nummuts = train_data.num_mut[(train_data.labels == 1).squeeze(-1)]

# Euclidean distance classifier  -------------------------------
classifier = DistanceBasedClassifier(train_input)
classifier_guess = classifier.forward(inputs)
    

distances = get_dist_vec(inputs, classifier_guess)
print(distances)
distance_cutoff = 1.0

classification_results = (distances <= distance_cutoff).to(torch.int64).reshape(-1,1)
print(classification_results)

plot_metric_vs_mutations_classifier(classification_results, label_classifier, num_mut, plot_path = 'test_ae_classifier/detector_performance_euclidean_1.png')

plot_roc_and_hist_nummut(inputs, classifier_guess, label_classifier, num_mut, plot_path = 'test_ae_classifier/nummut_euclidean', show=False)
