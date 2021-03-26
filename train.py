import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SignatureNet
from utilities import plot_signature, get_data_batches, get_entropy, confusion_matrix_plot

# Model params
num_hidden_layers = 4
num_neurons = 500
num_classes = 72
intial_learning_rate = 0.0001
learning_rate_steps = 500
learning_rate_gamma = 0.1

# Training params
experiment_id = "test_110"
iterations = 1e3
batch_size = 50
num_samples = 5000

if __name__ == "__main__":
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:,i]).type(torch.float32) for i in range(2, 74)]

    # TODO: Remove this line
    signatures = signatures[:num_classes]  # Classify only first 5 signatures

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))

    sn = SignatureNet(num_classes=num_classes, num_hidden_layers=num_hidden_layers, num_units=num_neurons)
    optimizer = optim.Adam(sn.parameters(), lr = intial_learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_steps, gamma=learning_rate_gamma)
    loss = nn.CrossEntropyLoss()
    
    predicted_list = torch.zeros(0,dtype=torch.long)
    label_list = torch.zeros(0,dtype=torch.long)

    for iteration in tqdm(range(int(iterations))):
        input_batch, label_batch = get_data_batches(signatures=signatures,
                                                    batch_size=batch_size,
                                                    n_samples=num_samples)
        optimizer.zero_grad()

        predicted_batch = sn(input_batch)

        if iteration > iterations/2:
            label_list = torch.cat([label_list,label_batch.view(-1)])
            predicted_list = torch.cat([predicted_list, torch.argmax(predicted_batch,1).view(-1)])

        l = loss(predicted_batch, label_batch)

        writer.add_scalar(f'loss', l.item(), iteration)
        #writer.add_scalar(f'entropy', get_entropy(predicted_batch), iteration)

        l.backward()
        optimizer.step()
        scheduler.step()
        #print(scheduler.get_lr())

    torch.save(sn.state_dict(), os.path.join("models", experiment_id))
    conf_mat = confusion_matrix_plot(label_list, predicted_list, range(num_classes))

    sm = torch.nn.Softmax()
    for i in range(num_classes):
        prediction = sn(signatures[i].unsqueeze(dim=0))
        probabilities = sm(prediction)
        print(i)
        print(probabilities)