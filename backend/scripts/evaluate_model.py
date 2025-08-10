import os
import torch
from torch import nn
from models.graph_vae import GraphVAE
from scripts.data_preparation import load_data

MODEL_PATH = os.path.join("models", "gvae_trained.pth")

def evaluate():
    print("📊 Starting model evaluation...")

    # Load data
    _, test_loader = load_data(split_ratio=0.8)

    # Initialize model
    model = GraphVAE()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            recon, mu, logvar = model(batch)
            loss = criterion(recon, batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"✅ Evaluation complete. Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    evaluate()
