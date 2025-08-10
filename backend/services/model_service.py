"""
Model service: load models, sample/generate, evaluate, RL stub.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# Import GraphVAE (could be DummyVAE if PyG missing)
from models.graph_vae import GraphVAE

# Import helpers from props (graph_to_smiles, smiles_to_graph might be placeholder)
from services.props import graph_to_smiles

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_models():
    """
    Load VAE and RL artifacts if present. Returns dict with keys 'vae' and 'rl'.
    If a trained VAE exists at models/gvae_trained.pth, try to load it.
    Otherwise instantiate an untrained GraphVAE (or DummyVAE).
    """
    vae = None
    try:
        # create model object
        vae = GraphVAE()
        model_path = MODEL_DIR / "gvae_trained.pth"
        if model_path.exists():
            try:
                state = torch.load(str(model_path), map_location="cpu")
                # If GraphVAE is a class instance, attempt to load state dict
                if hasattr(vae, "load_state_dict"):
                    vae.load_state_dict(state)
                # set eval if PyTorch module
                if hasattr(vae, "eval"):
                    vae.eval()
            except Exception:
                # If GraphVAE is dummy or mismatch, ignore
                pass
    except Exception:
        vae = GraphVAE()  # fallback

    # RL artifact stub (could be path to a saved agent)
    rl_agent = None
    rl_path = MODEL_DIR / "ppo_molgen.zip"
    if rl_path.exists():
        rl_agent = str(rl_path)

    return {"vae": vae, "rl": rl_agent}


def sample_from_latent(models: dict, z: Optional[list] = None, n: int = 1):
    """
    Generates molecules from a latent vector or random samples.
    Returns {"smiles": [...], "error": optional}
    """
    vae = models.get("vae")
    if vae is None:
        return {"smiles": ["CCO"] * n, "error": "VAE not available"}

    try:
        # Normalize input z handling
        if z is None:
            # Use num random samples
            if hasattr(vae, "generate"):
                result = vae.generate(z=None, num_samples=n)
            else:
                result = ["CCO"] * n
        else:
            # If z is list of vectors or a single vector
            if isinstance(z, list) or isinstance(z, np.ndarray):
                # if list of numbers -> single vector; if nested -> multiple
                arr = np.array(z)
                if arr.ndim == 1:
                    # one vector
                    if hasattr(vae, "generate"):
                        result = vae.generate(z=arr, num_samples=1)
                    else:
                        result = ["CCO"]
                else:
                    # multiple vectors
                    if hasattr(vae, "generate"):
                        result = vae.generate(z=arr, num_samples=arr.shape[0])
                    else:
                        result = ["CCO"] * arr.shape[0]
            else:
                # unsupported type: return fallback
                result = ["CCO"] * n

        # result may be either tuple (node_logits, edge_logits) or list of smiles (dummy)
        if isinstance(result, tuple):
            # decode each item -> try graph_to_smiles, otherwise placeholder
            node_logits, edge_logits = result
            smiles_out = []
            # node_logits could be a tensor of shape (batch, max_atoms, atom_types)
            batch = getattr(node_logits, "shape", None)
            b = batch[0] if batch else n
            for i in range(b):
                try:
                    smi = graph_to_smiles(node_logits[i], edge_logits[i])
                except Exception:
                    smi = "CCO"
                smiles_out.append(smi)
            return {"smiles": smiles_out}
        else:
            # assume list of smiles
            return {"smiles": list(result)}
    except Exception as e:
        return {"smiles": ["CCO"] * n, "error": str(e)}


def optimize_with_rl(models: dict, target: dict, total_timesteps: int = 20000):
    """
    Placeholder RL trainer. Replace with real RL pipeline (stable-baselines3 PPO etc).
    For now returns a summary dictionary.
    """
    # If you want to integrate stable-baselines, load agent and train here.
    # Keeping minimal stub so backend works without heavy RL packages.
    return {
        "status": "stubbed",
        "target": target,
        "timesteps": total_timesteps,
        "notes": "Integrate RL training or load saved RL agent to enable."
    }


def evaluate_batch(models: dict, n: int = 32):
    """
    Sample n molecules and compute a quick validity placeholder.
    Returns dict with count, smiles[] and validity_rate (placeholder).
    """
    res = sample_from_latent(models, z=None, n=n)
    smiles = res.get("smiles", [])
    # For a quick validity check, we mark as valid if RDKit can parse (services.props.calculate_properties will be helpful)
    from services.props import calculate_properties
    val_count = 0
    props = []
    for smi in smiles:
        p = calculate_properties(smi)
        props.append(p)
        if p.get("valid"):
            val_count += 1
    validity_rate = val_count / max(1, len(smiles))
    return {"count": len(smiles), "smiles": smiles, "validity_rate": validity_rate, "properties": props}
