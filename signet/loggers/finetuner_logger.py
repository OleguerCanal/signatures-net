import os

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from signet.utilities.metrics import get_MSE,\
    get_negative_cosine_similarity,\
    get_cross_entropy2,\
    get_kl_divergence,\
    get_jensen_shannon,\
    get_wasserstein_distance



class FinetunerLogger:
    def __init__(self):
        self.metrics = {
            # "mse": get_MSE,
            # "cos": get_negative_cosine_similarity,
            # "cross_ent": get_cross_entropy2,
            "KL": get_kl_divergence,
            # "JS": get_jensen_shannon,
            # "W": get_wasserstein_distance,
        }
        
    def log_bias(self, label, pred, split):
        plt.close("all")
        bias_figure = plt.figure()
        bias = torch.mean(pred - label, 0).detach().cpu().numpy()
        plt.ylim(-0.025, 0.025)
        plt.bar(list(range(bias.shape[-1])), bias, )
        wandb.log({f"{split}_bias_vector": wandb.Image(bias_figure)})
        wandb.log({f"{split}_bias": np.sum(np.abs(bias))})

    def log(self,
            train_loss,
            train_prediction,
            train_label,
            train_classification_metrics,
            val_loss,
            val_prediction,
            val_label,
            val_classification_metrics,
            step):

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        
        self.log_bias(label=train_label, pred=train_prediction, split="train")
        self.log_bias(label=val_label, pred=val_prediction, split="val")

        for metric_name in self.metrics.keys():
            metric = self.metrics[metric_name]
            wandb.log({"train_" + metric_name: metric(train_prediction, train_label).item()})
            wandb.log({"val_" + metric_name: metric(val_prediction, val_label).item()})

        for metric_name in train_classification_metrics.keys():
            try:
                wandb.log({"train_" + metric_name: train_classification_metrics[metric_name].item()})
            except:
                pass

        for metric_name in val_classification_metrics.keys():
            try:
                wandb.log({"val_" + metric_name: val_classification_metrics[metric_name].item()})
            except:
                pass
