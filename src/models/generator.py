import torch
from torch import nn

class Generator(nn.Module):
    
    def __init__(self,
                 input_size=72,
                 num_hidden_layers=2,
                 latent_dim=50,
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Generator"
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        encoder_layers = []
        for i in range(num_hidden_layers):
            in_features = int(input_size - (input_size-latent_dim)*i/num_hidden_layers)
            out_features = int(input_size - (input_size-latent_dim)*(i+1)/num_hidden_layers)
            print("in_features:", in_features, "out_features:", out_features)
            layer = nn.Linear(in_features, out_features)            
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(modules=encoder_layers)
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.var_layer = nn.Linear(latent_dim, latent_dim)

        decoder_layers = []
        for i in reversed(range(num_hidden_layers+1)):
            in_features = int(input_size - (input_size-latent_dim)*(i+1)/(num_hidden_layers + 1))
            out_features = int(input_size - (input_size-latent_dim)*i/(num_hidden_layers + 1))
            print("in_features:", in_features, "out_features:", out_features)
            layer = nn.Linear(in_features, out_features)            
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(modules=decoder_layers)
        self.activation = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        self.Normal = torch.distributions.Normal(0, 1)
        if device == "cuda":
            print("sending to cuda")
            self.Normal.loc = self.Normal.loc.cuda()
            self.Normal.scale = self.Normal.scale.cuda()


    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        z_mu = self.mean_layer(x)
        z_log_var = self.var_layer(x)
        z_std = torch.exp(0.5*z_log_var)
        return z_mu, z_std

    def decode(self, x):
        for layer in self.decoder_layers[:-1]:
            x = self.activation(layer(x))
        x = self.relu(self.decoder_layers[-1](x))
        x = x/x.sum(dim=1).reshape(-1,1)
        return x

    def forward(self, x, noise=True):
        z_mu, z_std = self.encode(x)
        if noise:
            # z = z_mu + z_std*self.Normal.sample(z_mu.shape)
            z = torch.randn(size = (z_mu.size(0), z_mu.size(1)))
            z = z_mu + z_std*z
        else:
            z = z_mu
        x = self.decode(z)
        return x, z_mu, z_std

    def generate(self, batch_size:int, std = 1.0):
        shape = tuple((batch_size, self.latent_dim))
        z = self.Normal.sample(shape)*std
        labels = self.decode(z)
        return labels

    def filter(self, syntethic_data, real_labels, quantile=0.75, print_dist_stats=False):
        """Remove outliers of synthetic_data by omitting
           (1 - quantile)% most-different points from the original dataset
        """
        def min_dist(point):
            return ((real_labels - point).pow(2)).mean(dim=1).min()
        distances = torch.tensor([min_dist(p) for p in syntethic_data])

        if print_dist_stats:
            print("Min dist:", distances.min())
            print("Mean dist:", distances.mean())
            print("Max dist:", distances.max())

        quantiles = torch.quantile(distances, torch.tensor([quantile]), keepdim=True)
        accepted = distances < quantiles.flatten()[-1].item()
        return syntethic_data[accepted, ...]