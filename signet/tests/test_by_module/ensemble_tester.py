import os
import sys

import torch
import numpy as np

from signet import DATA, TRAINED_MODELS
from signet.utilities.io import csv_to_tensor, read_model
from signet.modules.finetunner_ensemble import FinetunnerEnsemble

DEVICE = "cuda"

if __name__ == "__main__":
    input_batch = csv_to_tensor(os.path.join(DATA, "realistic_nummuts_data/test_large_input.csv"), device=DEVICE)
    baseline_batch = csv_to_tensor(os.path.join(DATA, "realistic_nummuts_data/test_large_baseline.csv"), device=DEVICE)
    label_batch = csv_to_tensor(os.path.join(DATA, "realistic_nummuts_data/test_large_label.csv"), device=DEVICE)
    
    model_ids = [
        "1_500_1", "1_500_2", "1_500_3",
        "2_500_1", "2_500_2", "2_500_3",
        "3_500_1", "3_500_2", "3_500_3",
        "2_1000_1", "2_1000_2", "2_1000_3",
        "1_layers_1000_neurons"
    ]
    models = [read_model(os.path.join(TRAINED_MODELS, "exp_ensemble", model_id), device=DEVICE) for model_id in model_ids]
    finetuner = FinetunnerEnsemble(finetuners=models)
    finetuner_guess = finetuner(mutation_dist=input_batch,
                                baseline_guess=baseline_batch,
                                num_mut=label_batch[:, -1].view(-1, 1))
    
    print(finetuner_guess.shape)

