import torch
from torch import nn

class VaeClassifier(nn.Module):
    def __init__(self, vae):
        super(VaeClassifier, self).__init__()
        self.vae = vae
        
    def forward(self, mutation_dist, num_mut=None):
        """nummut is not used in this model, but it is included for compatibility with other models
        """
        reconstruction = self.vae.forward(mutation_dist, noise=False)[0]
        return - torch.mean((reconstruction - mutation_dist) ** 2, dim=1)


if __name__ == "__main__":
    import os
    import logging

    from argparse import ArgumentParser
    import torch

    from signaturesnet import DATA, TRAINED_MODELS
    from signaturesnet.utilities.io import csv_to_tensor, read_model

    # Read data
    mutation_vectors = csv_to_tensor("/home/oleguer/software/SigNet/signaturesnet/data/datasets/example_input.csv", header=0, index_col=0)
    
    # Get the number of mutations for each mutational vector and normalize the vector
    num_muts = torch.sum(mutation_vectors, dim=1).reshape(-1, 1)
    normalized_mutation_vectors = mutation_vectors / num_muts

    # Run classification
    detector = read_model(os.path.join(TRAINED_MODELS, "detector"))
    classification_guess = detector(
        mutation_dist=normalized_mutation_vectors,
        num_mut=num_muts
    )

    # Classify results
    classification_cutoff = 0.5
    classification_results = (classification_guess >= classification_cutoff).to(torch.int64)
    
    print("Classifications:\n", classification_results)
    
    vae = read_model(os.path.join(TRAINED_MODELS, "generator"))  # NOTE this vae doesnt work, it must be trained on mutation_distro, not signature_distro
    vae_classifier = VaeClassifier(vae=vae)
    vae_classification_guesses = vae_classifier.forward(mutation_dist=normalized_mutation_vectors)

    print("here")
    