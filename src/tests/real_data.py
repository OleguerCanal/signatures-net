import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_signatures, tensor_to_csv
from utilities.io import read_signatures, read_test_data, read_model, csv_to_tensor
from modules.combined_finetuner import CombinedFinetuner
from models.baseline import Baseline

def read_real_data():
    inputs = data_folder + "real_data/PCAWG_data.csv"
    labels = data_folder + "real_data/sigprofiler_not_norm_PCAWG.csv"
    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
    labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)

    nummut = torch.sum(inputs, dim=1)
    # inputs = normalize_data(inputs,
                            # opportunities_name_or_path="../../data/real_data/3mer_WG_hg37.txt")
                            # opportunities_name_or_path="../../data/real_data/abundances_trinucleotides.txt")
                            # opportunities_name_or_path="../../data/real_data/norm_38.txt")
                            # opportunities_name_or_path="../../data/real_data/new_norm.txt")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)

    signatures = read_signatures("../../data/data.xlsx")
    baseline = Baseline(signatures)
    baselines = baseline.get_weights_batch(inputs)
    return inputs, baselines, labels, nummut

def read_synt_data():
    input_batch = csv_to_tensor("../../data/exp_not_norm/test_generator_input.csv")
    label_batch = csv_to_tensor("../../data/exp_not_norm/test_generator_label.csv")
    # baseline_batch = csv_to_tensor("../../data/exp_not_norm/test_generator_input.csv")
    signatures = read_signatures("../../data/data.xlsx")
    baseline = Baseline(signatures)
    baselines = baseline.get_weights_batch(input_batch)
    return input_batch, baselines, label_batch[:, :-1], label_batch[:, -1]

def read_finetuner():
    experiment_id = "exp_not_norm"
    models_path = "../../trained_models/%s/"%experiment_id
    finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_not_norm_no_baseline_low",
                                            large_mum_mut_dir=models_path + "finetuner_not_norm_large")
    return finetuner

def normalize(a, b):
    """Normalize 1 wrt b
    """
    a_mean = torch.mean(a, dim=0)
    b_mean = torch.mean(b, dim=0)
    return (a/a_mean)*b_mean

def small_to_unkown(a, thr = 0.01):
    """
    Small values to unknown category
    """
    b = a.detach().clone()
    b[b>=thr] = 0
    unknown = torch.sum(b, dim=1)
    a[a<thr] = 0
    return torch.cat([a, unknown.reshape(-1,1)], dim = 1)

if __name__=="__main__":
    data_folder = "../../data/"

    real_inputs, real_baseline, real_labels, real_nummut = read_real_data()
    synt_inputs, synt_baseline, synt_labels, synt_nummut = read_synt_data()

    # real_inputs_norm = normalize(real_inputs, synt_inputs)
    real_inputs_norm = real_inputs

    finetuner = read_finetuner()
    real_guess = finetuner(mutation_dist=real_inputs_norm, baseline_guess=real_baseline, num_mut=real_nummut)
    synt_guess = finetuner(mutation_dist=synt_inputs, baseline_guess=synt_baseline, num_mut=synt_nummut)

    real_labels_unknown = small_to_unkown(real_labels)
    synt_labels_unknown = small_to_unkown(synt_labels)
    real_guess_unknown = small_to_unkown(real_guess)
    synt_guess_unknown = small_to_unkown(synt_guess)
    real_baseline_unknown = small_to_unkown(real_baseline)
    tensor_to_csv(real_baseline, "../../data/real_data/baseline_signet.csv")
    tensor_to_csv(real_guess_unknown, "../../data/real_data/real_data_signet.csv")

    signatures = read_signatures(data_folder + "data.xlsx")
    real_label_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(real_labels)))
    real_guess_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(real_guess)))
    synt_label_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(synt_labels)))
    synt_guess_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(synt_guess)))


    # synt_inputs = synt_inputs[synt_nummut > 1e3]
    # synt_labels = synt_labels[synt_nummut > 1e3]
    # synt_guess = synt_guess[synt_nummut > 1e3]
    # synt_label_rec = synt_label_rec[synt_nummut > 1e3]
    # synt_guess_rec = synt_guess_rec[synt_nummut > 1e3]


    # print("MSE weights")
    # print(get_MSE(synt_labels_unknown, synt_guess_unknown))
    # print(get_MSE(real_labels_unknown, real_guess_unknown))

    # print("MSE reconstruction")
    # print(get_MSE(real_inputs, real_label_rec))
    # print(get_MSE(real_inputs, real_guess_rec))

    # print("Cosine Similarity")
    # print(get_cosine_similarity(real_inputs, real_label_rec))
    # print(get_cosine_similarity(real_inputs, real_guess_rec))
    data = {
             "synt_labels": synt_labels_unknown,
             "synt_guess": synt_guess_unknown,
             "real_labels": real_labels_unknown,
             "real_guess": real_guess_unknown,
             "baseline_guess": real_baseline_unknown,
             }
    # plot_bars(data, max=73)

    data = {
            # "synt_inputs": synt_inputs,
            # "synt_label_rec": synt_label_rec,
            # "synt_guess_rec": synt_guess_rec,
            "real_inputs": real_inputs,
            # "real_inputs_norm": real_inputs_norm,
            "real_label_rec": real_label_rec,
            "real_guess_rec": real_guess_rec,
            }
    # plot_bars(data)

    def boxplots(real_guess, real_labels, num_sigs_range = [0,36], only_present = False, legend_names = ['SigNet real', 'SigProfiler Labels'] ):
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        if only_present:
            real_guess_changed = real_guess.detach().clone()
            real_labels_changed = real_labels.detach().clone()

            real_guess_changed = real_guess_changed.detach().numpy()
            real_guess_changed = np.where(real_guess_changed>0.01, real_guess_changed, np.nan)
            real_labels_changed = real_labels_changed.detach().numpy()
            real_labels_changed = np.where(real_labels_changed>0.01, real_labels_changed, np.nan)
            
            mask = ~np.isnan(real_guess_changed)
            real_guess_changed = [d[m] for d, m in zip(real_guess_changed.T, mask.T)]
            mask = ~np.isnan(real_labels_changed)
            real_labels_changed = [d[m] for d, m in zip(real_labels_changed.T, mask.T)]
            fig = plt.figure()
            axs = plt.axes()

            axs.boxplot(real_guess_changed[num_sigs_range[0]:num_sigs_range[1]], positions = np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2, widths = 0.1)
            axs.boxplot(real_labels_changed[num_sigs_range[0]:num_sigs_range[1]], positions = np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2+1, widths = 0.1)

            axs.set_xticks(np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2+0.5)
            axs.set_xticklabels(list(pd.read_excel("../../data/data.xlsx").columns)[(num_sigs_range[0]+1):(num_sigs_range[1]+1)], rotation=90)
            # plt.show()
        else:
            fig = plt.figure()
            axs = plt.axes()

            real_guess_changed = real_guess.detach().clone()
            real_labels_changed = real_labels.detach().clone()

            real_guess_changed[real_guess_changed<0.01] = 0
            real_guess_changed[real_guess_changed>=0.01] = 1
            prop_tumors_real = torch.sum(real_guess_changed, dim=0)/real_guess_changed.shape[0] 

            real_labels_changed[real_labels_changed<0.01] = 0
            real_labels_changed[real_labels_changed>=0.01] = 1
            prop_tumors_labels = torch.sum(real_labels_changed, dim=0)/real_labels_changed.shape[0] 

            axs.bar(np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2, prop_tumors_real[num_sigs_range[0]:num_sigs_range[1]])
            axs.bar(np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2+1, prop_tumors_labels[num_sigs_range[0]:num_sigs_range[1]])
            plt.legend(legend_names)
            axs.boxplot(torch.transpose(real_guess[:,num_sigs_range[0]:num_sigs_range[1]],1,0), positions = np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2, widths = 0.1)
            axs.boxplot(torch.transpose(real_labels[:,num_sigs_range[0]:num_sigs_range[1]],1,0), positions = np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2+1, widths = 0.1)

            axs.set_xticks(np.array(range(num_sigs_range[1]-num_sigs_range[0]))*2+0.5)
            axs.set_xticklabels(list(pd.read_excel("../../data/data.xlsx").columns)[(num_sigs_range[0]+1):(num_sigs_range[1]+1)], rotation=90)
            # plt.show()

    # boxplots(real_guess, real_labels, num_sigs_range = [0,36], only_present = False, legend_names = ['SigNet real', 'SigProfiler Labels'])
    # boxplots(real_guess, real_labels, num_sigs_range = [36,72], only_present = False, legend_names = ['SigNet real', 'SigProfiler Labels'])
    # boxplots(real_guess, real_labels, num_sigs_range = [0,36], only_present = True)
    # boxplots(real_guess, real_labels, num_sigs_range = [36,72], only_present = True)

    # boxplots(synt_guess, synt_labels, num_sigs_range = [0,36], only_present = False, legend_names = ['SigNet Synthetic', 'Synthetic Labels'])
    # boxplots(synt_guess, synt_labels, num_sigs_range = [36,72], only_present = False, legend_names = ['SigNet Synthetic', 'Synthetic Labels'])
    # boxplots(synt_guess, synt_labels, num_sigs_range = [0,36], only_present = True)
    # boxplots(synt_guess, synt_labels, num_sigs_range = [36,72], only_present = True)

    # boxplots(real_labels, synt_labels, num_sigs_range = [0,36], only_present = False, legend_names = ['SigProfiler Labels', 'Synthetic Labels'])
    # boxplots(real_labels, synt_labels, num_sigs_range = [36,72], only_present = False, legend_names = ['SigProfiler Labels', 'Synthetic Labels'])
    # boxplots(real_labels, synt_labels, num_sigs_range = [0,36], only_present = True)
    # boxplots(real_labels, synt_labels, num_sigs_range = [36,72], only_present = True)


    generator = read_model("../../trained_models/exp_not_norm/generator_best_3", device="cpu")
    examples = generator.generate(10000, std = 1).detach()
    import matplotlib.pyplot as plt

    boxplots(real_labels, examples, num_sigs_range = [0,36], only_present = False, legend_names = ['SigProfiler Labels', 'Synthetic Labels'])
    boxplots(real_labels, examples, num_sigs_range = [36,72], only_present = False, legend_names = ['SigProfiler Labels', 'Synthetic Labels'])
    boxplots(real_labels, examples, num_sigs_range = [0,36], only_present = True)
    boxplots(real_labels, examples, num_sigs_range = [36,72], only_present = True)
    plt.show()
