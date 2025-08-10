"""
Simple Gym environment that wraps the generator's latent vector space.
This is a minimal env to demo RL integration. Replace with your full environment
and proper action/observation definitions as needed.
"""

import numpy as np
from typing import Tuple
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    # minimal fallback if gymnasium not available
    gym = None
    spaces = None

from services.props import calculate_properties, graph_to_smiles
import torch

class LatentMolEnv:
    """A tiny stand-in environment (non-gym) if gym not installed."""
    def __init__(self, generator, latent_dim=64):
        self.generator = generator
        self.latent_dim = latent_dim
        self.state = np.zeros(latent_dim, dtype=np.float32)

    def reset(self):
        self.state = np.random.randn(self.latent_dim).astype(np.float32)
        return self.state

    def step(self, action: np.ndarray):
        self.state = np.clip(self.state + action * 0.1, -3.0, 3.0)
        # generator may accept torch tensor or numpy
        try:
            out = self.generator.generate(z=torch.tensor([self.state], dtype=torch.float32))
            if isinstance(out, tuple):
                # (node_logits, edge_logits) -> decode
                smi = graph_to_smiles(out[0][0], out[1][0])
            else:
                smi = out[0] if isinstance(out, (list, tuple)) else "CCO"
        except Exception:
            smi = "CCO"
        props = calculate_properties(smi)
        reward = props.get("qed", 0.0) - max(0, props.get("sa", 0) - 6) * 0.1
        done = True  # episodes of one-step for simplicity
        info = {"smiles": smi, "properties": props}
        return self.state, reward, done, info

# If gym is installed, provide a gym.Env wrapper
if gym is not None:
    class MolecularEnv(gym.Env):
        def __init__(self, generator_model, latent_dim=64):
            super().__init__()
            self.generator = generator_model
            self.latent_dim = latent_dim
            self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(latent_dim,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)
            self.current_latent = None

        def reset(self, seed=None, options=None):
            self.current_latent = np.random.randn(self.latent_dim).astype(np.float32)
            return self.current_latent, {}

        def step(self, action):
            self.current_latent = np.clip(self.current_latent + action * 0.1, -3.0, 3.0).astype(np.float32)
            try:
                out = self.generator.generate(z=torch.tensor([self.current_latent], dtype=torch.float32))
                if isinstance(out, tuple):
                    smi = graph_to_smiles(out[0][0], out[1][0])
                else:
                    smi = out[0] if isinstance(out, (list, tuple)) else "CCO"
                props = calculate_properties(smi)
                reward = props.get("qed", 0.0) - max(0, props.get("sa", 0) - 6) * 0.1
            except Exception:
                smi = "CCO"
                props = {"valid": False}
                reward = -1.0
            terminated = True
            truncated = False
            info = {"smiles": smi, "properties": props}
            return self.current_latent, reward, terminated, truncated, info
