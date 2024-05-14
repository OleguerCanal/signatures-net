import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signet import DATA
from models import Baseline
from signet.utilities.data_generator import DataGenerator
from signet.utilities.data_partitions import DataPartitions
from signet.utilities.io import csv_to_pandas, read_signatures, csv_to_tensor
from signet.utilities.oversampler import CancerTypeOverSampler


def read_data_nummutnet(path, device):
    # TODO read input and labels
    train_input = csv_to_tensor(file=path + "/train_input.csv", device=device)
    train_label = csv_to_tensor(file=path + "/train_label.csv", device=device).to(torch.long)
    val_input = csv_to_tensor(file=path + "/val_input.csv", device=device)
    val_label = csv_to_tensor(file=path + "/val_label.csv", device=device).to(torch.long)

    train_data = DataPartitions(inputs=train_input,
                                labels=train_label,
                                extract_nummut=False)
    val_data = DataPartitions(inputs=val_input,
                              labels=val_label,
                              extract_nummut=False)
    return train_data, val_data

def generate_realistic_nummut_data(
                                    n_samples,
                                    data_id="real_data",
                                    data_folder=DATA,
                                    split="train",
                                    large_or_low="large",
                                    device="cuda",):
    # Read real weight combinations
    labels_filepath = os.path.join(data_folder, data_id, "PCAWG_sigProfiler_SBS_signatures_in_samples_v3.csv")
    df = csv_to_pandas(labels_filepath, header=0, index_col=0)
    weights = torch.from_numpy(df.iloc[:, 2:].values)
    # Add zeros for last 7 missing signatures
    weights = torch.cat([weights, torch.zeros(weights.size(0), 7).to(weights)], dim=1)
    nummuts = torch.sum(weights, axis=1).to(torch.float32).unsqueeze(1)

    # Create inputs associated to the labels:
    signatures = read_signatures(
        os.path.join(DATA, "data.xlsx"),
        mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx"))
    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   normalize=True,
                                   shuffle=True)
    inputs, labels = data_generator.make_input(labels=weights,
                                                split=split,
                                                large_low=large_or_low,
                                                nummuts=nummuts)
    inputs = inputs[:n_samples]
    labels = labels[:n_samples]

    # Run Baseline
    baseline = Baseline(signatures)
    train_baseline = baseline.get_weights_batch(inputs, n_workers=4)
    
    # Split into train and validation
    data = DataPartitions(inputs=inputs.float(),
                          prev_guess=train_baseline.float(),
                          labels=labels.float())
    data.to(device)
    return data


# def read_data_final_finetuner(device, data_id, data_folder=DATA, network_type="low"):
#     '''
#     Read all real data, oversample and generate samples with different numbers of mutations
#     to train the final finetuner.
#     '''
#     data_folder = data_folder + data_id
#     real_data = csv_to_pandas(data_folder + "/sigprofiler_not_norm_PCAWG.csv",
#                                     device=device, header=0, index_col=0,
#                                     type_df=data_folder + "/PCAWG_sigProfiler_SBS_signatures_in_samples_v3.csv")
#     ctypes = torch.tensor(real_data.values[:,-1], dtype=torch.float)
#     real_data = torch.tensor(real_data.values[:,:-1], dtype=torch.float)
#     real_data = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)
#     oversampler = CancerTypeOverSampler(real_data, ctypes)
#     oversampled_weights = oversampler.get_N_oversampled_set(N_samples=1)

#     # Create inputs associated to the labels:
#     signatures = read_signatures(
#         DATA + "data.xlsx",
#         mutation_type_order=DATA + "mutation_type_order.xlsx")
#     data_generator = DataGenerator(signatures=signatures,
#                                    seed=None,
#                                    shuffle=True)

#     train_input, train_label = data_generator.make_input(
#         labels=oversampled_weights,
#         split="train",
#         large_low=network_type,
#     )
    
#     # Run Baseline
#     sf = Baseline(signatures)
#     train_baseline = sf.get_weights_batch(train_input, n_workers=2)
    
#     train_data = DataPartitions(inputs=train_input.float().to(device),
#                                 prev_guess=train_baseline.float().to(device),
#                                 labels=train_label.float().to(device))
#     val_data = train_data
#     return train_data, val_data


def _shuffle(data):
    indexes = torch.randperm(data.shape[0])
    return data[indexes, ...]

def _convert_to_input(data_weights, network_type, split, device):
    signatures = read_signatures(
        file=DATA+"data.xlsx",
        mutation_type_order=DATA+"mutation_type_order.xlsx")
    
    data_generator = DataGenerator(
        signatures=signatures,
        seed=0,
        shuffle=True
    )
    
    inputs, labels = data_generator.make_input(
        labels=data_weights,
        split=split,
        large_low=network_type,
    )
    
    # Compute baselines
    sf = Baseline(signatures)
    baselines = sf.get_weights_batch(inputs, n_workers=6)

    data = DataPartitions(
        inputs=inputs,
        prev_guess=baselines,
        labels=labels,
    )
    data.to(device)
    return data

def read_finetuner_data(data_id, network_type, device):
    # Read real weight combinations (labels)
    real_data = csv_to_pandas(
        file=DATA + data_id + "/sigprofiler_not_norm_PCAWG.csv",
        device=device,
        header=0,
        index_col=0,
        type_df=DATA + data_id + "/PCAWG_sigProfiler_SBS_signatures_in_samples_v3.csv"
    )
    
    # Extract cancer types, remove cancertype column from input, add missing signatures
    ctypes = torch.tensor(real_data.values[:,-1], dtype=torch.float)
    real_data = torch.tensor(real_data.values[:,:-1], dtype=torch.float)
    real_data = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)
    
    # Oversample according to cancer type (if N_samples=0 nothing is done)
    oversampler = CancerTypeOverSampler(
        data=real_data,
        cancer_types=ctypes
    )
    oversampled_weights = oversampler.get_N_oversampled_set(N_samples=1)
                                                            
    # Shuffle data and split into train and validation
    oversampled_weights =_shuffle(oversampled_weights)
    n = oversampled_weights.shape[0]
    train_weights = oversampled_weights[:int(0.8*n)]
    val_weights = oversampled_weights[int(0.8*n):]
    
    train_data = _convert_to_input(
        data_weights=train_weights,
        network_type=network_type,
        split="train",
        device=device,
    )
    val_data = _convert_to_input(
        data_weights=val_weights,
        network_type=network_type,
        split="val",
        device=device,
    )
    return train_data, val_data
    
if __name__=="__main__":

    train_data, val_data = read_finetuner_data(
            device="cpu",
            data_id="real_data",
            network_type="large"
        )
    