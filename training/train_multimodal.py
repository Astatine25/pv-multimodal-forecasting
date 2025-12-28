import torch
from torch.utils.data import DataLoader, TensorDataset
from models.transformer import MultimodalTransformer

def main():
    # Dummy data (replace with real tensors later)
    B, T = 32, 12
    weather_dim = 6
    img_dim = 128

    weather = torch.randn(B, T, weather_dim)
    images = torch.randn(B, T, img_dim)
    y = torch.randn(B, 1)

    dataset = TensorDataset(weather, images, y)
    loader = DataLoader(dataset, batch_size=8)

    model = MultimodalTransformer(
        weather_dim=weather_dim,
        img_embed_dim=img_dim
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(5):
        total_loss = 0
        for w, i, target in loader:
            optimizer.zero_grad()
            pred = model(w, i)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
