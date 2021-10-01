import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classified_tunning import ClassifiedFinetuner
from models.baseline import Baseline
from utilities.io import csv_to_tensor, read_signatures, read_test_data
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs

experiment_id = "exp_random_2_nets"

model_path = "../../trained_models/"

classifier_exp = "exp_classifier"
classifier_model = model_path + '/' + classifier_exp + '/' + "classifier_baseline"

random_exp = "exp_0"
random_finetuner_model = model_path + "finetuner_random"
realistic_finetuner_model = model_path + "finetuner_realistic"

experiment_id = "exp_classifier"

input_batch = csv_to_tensor("../../data/" + experiment_id + "/test_input.csv", device='cpu')
num_mut = csv_to_tensor("../../data/" + experiment_id + "/test_num_mut.csv", device='cpu')
label = csv_to_tensor("../../data/" + experiment_id + "/test_label.csv", device='cpu')
signatures = read_signatures("../../data/data.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

finetuner = ClassifiedFinetuner(classifier_model,
                 realistic_finetuner_model,
                 random_finetuner_model)

finetuner_guess, ind_order = finetuner(input_batch, baseline_guess, num_mut)
baseline_guess = baseline_guess[ind_order,:]

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess]

list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_muts.png"%experiment_id)

list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_muts.png"%experiment_id)
