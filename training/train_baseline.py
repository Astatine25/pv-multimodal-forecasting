import torch

def train_lstm(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
