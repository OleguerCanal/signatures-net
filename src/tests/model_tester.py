import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sn


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_test_data
from utilities.plotting import plot_signature, plot_confusion_matrix, plot_weights, plot_weights_comparison, plot_interval_performance
from utilities.metrics import *
from utilities.dataloader import DataLoader
from models.signature_finder import SignatureFinder
from models.finetuner import FineTuner
from models.error_finder import ErrorFinder

class ModelTester:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def test(self, guessed_labels, true_labels):
        mse = get_MSE(guessed_labels, true_labels)
        cross_entropy = get_cross_entropy(guessed_labels, true_labels)
        kl_divergence = get_kl_divergence(guessed_labels, true_labels)
        js_divergence = get_jensen_shannon(guessed_labels, true_labels)
        cosine_similarity = get_cosine_similarity(guessed_labels, true_labels)
        wasserstein_distance = get_wasserstein_distance(
            guessed_labels, true_labels)

        print("mse:", np.round(mse.item(), decimals=5))
        print("cross_entropy:", np.round(cross_entropy.item(), decimals=5))
        print("kl_divergence:", np.round(kl_divergence.item(), decimals=5))
        print("js_divergence:", np.round(js_divergence.item(), decimals=5))
        print("cosine_similarity:", np.round(
            cosine_similarity.item(), decimals=3))
        print("wasserstein_distance:", np.round(
            wasserstein_distance.item(), decimals=3))

        label_sigs, predicted_sigs = self.probs_batch_to_sigs(
            true_labels, guessed_labels, cutoff=0.05, num_classes=self.num_classes)

        conf_mat = plot_confusion_matrix(
            label_sigs, predicted_sigs, range(self.num_classes+1))

    def probs_batch_to_sigs(self, label_batch, predicted_batch, cutoff, num_classes):
        label_sigs_list = torch.zeros(0, dtype=torch.long)
        predicted_sigs_list = torch.zeros(0, dtype=torch.long)
        for i in range(len(label_batch)):
            for j in range(len(label_batch[i])):
                if label_batch[i][j] > cutoff and predicted_batch[i][j] > cutoff:
                    label_sigs_list = torch.cat(
                        [label_sigs_list, torch.from_numpy(np.array([j]))])
                    predicted_sigs_list = torch.cat(
                        [predicted_sigs_list, torch.from_numpy(np.array([j]))])
                if label_batch[i][j] > cutoff and predicted_batch[i][j] < cutoff:
                    label_sigs_list = torch.cat(
                        [label_sigs_list, torch.from_numpy(np.array([j]))])
                    predicted_sigs_list = torch.cat(
                        [predicted_sigs_list, torch.from_numpy(np.array([num_classes]))])
                if label_batch[i][j] < cutoff and predicted_batch[i][j] > cutoff:
                    label_sigs_list = torch.cat(
                        [label_sigs_list, torch.from_numpy(np.array([num_classes]))])
                    predicted_sigs_list = torch.cat(
                        [predicted_sigs_list, torch.from_numpy(np.array([j]))])
        return label_sigs_list, predicted_sigs_list


if __name__ == "__main__":
    # Model params finetuner
    experiment_id_finetune = "realistic_data_finetuner_1"
    num_hidden_layers = 1
    num_neurons = 1500
    num_classes = 72

    # Model params error
    experiment_id_error_learner = "realistic_data_error_finder_1"
    num_hidden_layers_pos = 2
    num_neurons_pos = 500
    num_hidden_layers_neg = 2
    num_neurons_neg = 500
    normalize_mut = 2e3

    # experiment_id_error_learner = "error_finder_model_new_loss"
    # num_hidden_layers_pos = 5
    # num_neurons_pos = 1000
    # num_hidden_layers_neg = 5
    # num_neurons_neg = 1000
    # normalize_mut = 2e4

    # Generate data
    data = pd.read_excel("../../data/data.xlsx")
    # signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
    #               for i in range(2, 74)][:num_classes]

    # input_batch = torch.tensor(pd.read_csv(
    #     "../../data/test_input_w01.csv", header=None).values, dtype=torch.float)
    # label_mut_batch = torch.tensor(pd.read_csv(
    #     "../../data/test_label_w01.csv", header=None).values, dtype=torch.float)
    # label_batch = label_mut_batch[:, :num_classes]
    # num_mut = torch.reshape(
    #     label_mut_batch[:, num_classes], (list(label_mut_batch.size())[0], 1))

    # baseline_batch = torch.tensor(pd.read_csv(
    #     "../../data/test_w01_baseline_JS.csv", header=None).values, dtype=torch.float)

    device = torch.device("cpu")
    input_batch, baseline_batch, label_mut_batch = read_test_data(device, data_folder="../../data", realistic_data = True, num_classes=72)
    num_mut = torch.reshape(label_mut_batch[:, num_classes], (list(label_mut_batch.size())[0], 1))
    label_batch = label_mut_batch[:, :num_classes]
    
    # Instantiate model and do predictions for finetuner:
    model = FineTuner(num_classes=num_classes,
                      num_hidden_layers=num_hidden_layers,
                      num_units=num_neurons)
    model.load_state_dict(torch.load(os.path.join(
        "../../trained_models", experiment_id_finetune), map_location=torch.device('cpu')))
    model.eval()
    guessed_labels = model(input_batch, baseline_batch)

    # Instantiate model and do predictions for error learner:
    model_error = ErrorFinder(num_classes=num_classes,
                              num_hidden_layers_pos=num_hidden_layers_pos,
                              num_units_pos=num_neurons_pos,
                              num_hidden_layers_neg=num_hidden_layers_neg,
                              num_units_neg=num_neurons_neg,
                              normalize_mut=normalize_mut)

    model_error.load_state_dict(torch.load(os.path.join(
        "../../trained_models", experiment_id_error_learner), map_location=torch.device('cpu')))
    model_error.eval()
    pred_upper, pred_lower = model_error(guessed_labels, num_mut)

    # Get metrics
    model_tester = ModelTester(num_classes=num_classes)
    model_tester.test(guessed_labels=guessed_labels, true_labels=label_batch)

    # False negatives:
    label_sigs_list, predicted_sigs_list = model_tester.probs_batch_to_sigs(
        label_batch[:, :72], guessed_labels, 0.05, num_classes)
    FN = sum(predicted_sigs_list == num_classes)
    print('Number of FN:', FN)

    # False positives:
    FP = sum(label_sigs_list == num_classes)
    print('Number of FP:', FP)

    # Plot signatures
    plot_weights_comparison(label_batch[0, :].detach().numpy(), guessed_labels[0, :].detach().numpy(
    ), pred_upper[0, :].detach().numpy(),pred_lower[0, :].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(label_batch[9, :].detach().numpy(), guessed_labels[9, :].detach().numpy(
    ), pred_upper[9, :].detach().numpy(),pred_lower[9, :].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(label_batch[22, :].detach().numpy(), guessed_labels[22, :].detach().numpy(
    ), pred_upper[22, :].detach().numpy(),pred_lower[22, :].detach().numpy(), list(data.columns)[2:])

    # Plot interval performance
    plot_interval_performance(label_batch, pred_upper,pred_lower, list(data.columns)[2:])

    # Print metrics
    in_prop, mean_length = get_pi_metrics(label_batch, pred_lower=pred_lower, pred_upper=pred_upper)
    print("In proportion:", in_prop.detach().numpy())
    print("Mean length:", mean_length.detach().numpy())
