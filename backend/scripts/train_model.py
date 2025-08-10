import os
import torch
from torch import nn, optim
from models.graph_vae import GraphVAE  # Import your model class
from scripts.data_preparation import load_data  # Import your data loader function

# Training Config
EPOCHS = 50
LEARNING_RATE = 1e-3
SAVE_PATH = os.path.join("models", "gvae_trained.pth")

def train():
    print("🚀 Starting model training...")

    # Load data
    train_loader, _ = load_data(split_ratio=0.8)

    # Initialize model, optimizer, and loss function
    model = GraphVAE()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = criterion(recon, batch)  # Modify based on your loss function
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()
