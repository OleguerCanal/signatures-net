import copy
import os
import random
import sys

import pandas as pd
import torch

from signet import DATA
from signet.modules.signet_module import SigNet
from signet.utilities.io import read_methods_guesses
from signet.utilities.plotting import final_plot_all_metrics_vs_mutations, final_plot_interval_metrics_vs_mutations, plot_metric_vs_mutations_classifier, plot_time_vs_mutations

import musical

def subsample(n, inputs, labels):
    indexes = list(range(inputs.shape[0]))
    random.shuffle(indexes)
    ind_list = indexes[:n]
    inputs = inputs.iloc[ind_list]
    labels = labels.iloc[ind_list]
    return inputs, labels

def get_musical_guess(inputs, signatures):
    musical_input = copy.deepcopy(inputs).transpose()
    musical_signatures = copy.deepcopy(signatures)
    musical_signatures = musical_signatures.set_index("Type")
    musical_input.index = musical_signatures.index

    musical_guess, _ = musical.refit.refit(musical_input, musical_signatures, method='likelihood_bidirectional', thresh=0.001)
    musical_guess.transpose().to_csv("musical_guess.csv")

    musical_guess = torch.from_numpy(musical_guess.transpose().values).to(torch.float)
    return musical_guess


# Load data
data_path = DATA + "/datasets/"
inputs = pd.read_csv(data_path + "test_input.csv", header=None, index_col=None)
labels = pd.read_csv(data_path + "test_label.csv", header=None, index_col=None)

# inputs, labels = subsample(1000, inputs, labels)

num_mut = labels[72]
print("data loaded")

signatures = pd.read_excel(DATA + '/data.xlsx')

musical_guess = get_musical_guess(inputs, signatures)


input_df = inputs.mul(num_mut,0)
input_df.index = ['sample_' + str(i) for i in list(range(len(input_df.index)))]
input_df.columns = signatures['Type']

# Load model
path = "../../trained_models/"
signet = SigNet(classifier=path + "detector",
                finetuner_realistic_low=path + "finetuner_low",
                finetuner_realistic_large=path + "finetuner_large",
                errorfinder=path + "errorfinder",
                opportunities_name_or_path=None,
                signatures_path=DATA + "/data.xlsx",
                mutation_type_order=DATA + "/mutation_type_order.xlsx")

print("model read")

result = signet(input_df, numpy=False)
print("forwarded")

finetuner_guess, lower_bound, upper_bound, classification, normalized_input = result.get_output(format="tensor")

# list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA"]
# list_of_guesses, label = read_methods_guesses('cpu', "exp_all", list_of_methods, data_folder=DATA + "/")

list_of_methods = ['NNLS', 'Finetuner', 'musical']
list_of_guesses = [signet.baseline_guess, finetuner_guess[:,:-1], musical_guess]

labels = torch.tensor(labels.values, dtype=torch.float)
final_plot_all_metrics_vs_mutations(list_of_methods=list_of_methods,
                                    list_of_guesses=list_of_guesses,
                                    label=labels,
                                    signatures=signet.signatures,
                                    mutation_distributions=torch.tensor(inputs.values, dtype=torch.float),
                                    folder_path="../../../plots/paper/")

# classification_cutoff = 0.5
# classification_results = (classification >= classification_cutoff).to(torch.int64)
# plot_metric_vs_mutations_classifier(guess=classification_results,
#                                     label=torch.ones((inputs.shape[0])).to(torch.int64),
#                                     num_muts_list=labels[:, -1], 
#                                     plot_path="../../../plots/paper/")

# times = pd.read_csv(DATA + '/exp_all/other_methods/times/all_times.csv', index_col=0)
# num_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5]
# plot_time_vs_mutations(times, num_muts, plot_path="../../../plots/paper/", show=True)

# final_plot_interval_metrics_vs_mutations(labels, upper_bound, lower_bound, list(pd.read_excel(DATA + "/data.xlsx").columns)[1:], plot_path="../../../plots/paper/", show=False)
