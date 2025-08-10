"""
Graph VAE model file.
If torch_geometric is available it defines a GraphVAE using GCNConv.
If not available, it provides a DummyVAE with a simple decoder that returns placeholder SMILES.
"""

import torch
import torch.nn as nn
import numpy as np

# Try to import torch_geometric components; if unavailable, fall back to DummyVAE
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except Exception:
    HAS_PYG = False

if HAS_PYG:
    import torch.nn.functional as F

    class GraphEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.mean_layer = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x, edge_index, batch):
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            h_graph = global_mean_pool(h, batch)
            mu = self.mean_layer(h_graph)
            logvar = self.logvar_layer(h_graph)
            return mu, logvar

    class GraphDecoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim, max_atoms=50, atom_types=10):
            super().__init__()
            self.node_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_atoms * atom_types)
            )
            self.edge_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_atoms * max_atoms * 4)
            )
            self.max_atoms = max_atoms
            self.atom_types = atom_types

        def forward(self, z):
            batch = z.size(0)
            node_logits = self.node_head(z).view(batch, self.max_atoms, self.atom_types)
            edge_logits = self.edge_head(z).view(batch, self.max_atoms, self.max_atoms, 4)
            return node_logits, edge_logits

    class GraphVAE(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=128, latent_dim=64, max_atoms=50):
            super().__init__()
            self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim)
            self.decoder = GraphDecoder(latent_dim, hidden_dim, max_atoms=max_atoms)
            self.latent_dim = latent_dim

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, data):
            mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
            z = self.reparameterize(mu, logvar)
            node_logits, edge_logits = self.decoder(z)
            return node_logits, edge_logits, mu, logvar

        def generate(self, z=None, num_samples=1):
            if z is None:
                z = torch.randn(num_samples, self.latent_dim)
            elif isinstance(z, list) or isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
            with torch.no_grad():
                node_logits, edge_logits = self.decoder(z)
            return node_logits, edge_logits

else:
    # Dummy fallback
    class DummyVAE:
        def __init__(self, latent_dim=16):
            self.latent_dim = latent_dim

        def generate(self, z=None, num_samples=1):
            # Accept various z formats
            if z is None:
                n = num_samples
            else:
                if isinstance(z, list) or isinstance(z, np.ndarray):
                    arr = np.array(z)
                    n = arr.shape[0] if arr.ndim > 1 else 1
                elif hasattr(z, "shape"):
                    n = int(z.shape[0])
                else:
                    n = num_samples
            # Return simple placeholders
            smiles = ["CCO"] * n
            return smiles

    # Expose GraphVAE alias to the rest of code
    GraphVAE = DummyVAE
