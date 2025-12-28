import torch
from torch.utils.data import DataLoader
from models.transformer import MultimodalTransformer

def train_multimodal(model, loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

def main():
    # dummy example — replace with real dataset
    model = MultimodalTransformer(
        num_features=10,
        image_dim=128,
        hidden_dim=256
    )

    dummy_x = torch.randn(64, 10)
    dummy_y = torch.randn(64, 1)
    loader = DataLoader(list(zip(dummy_x, dummy_y)), batch_size=8)

    train_multimodal(model, loader)

if __name__ == "__main__":
    main()
