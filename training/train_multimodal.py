def train_multimodal(model, cnn, loader, optimizer, loss_fn):
    model.train()
    cnn.train()
    total_loss = 0

    for ts, img, y in loader:
        optimizer.zero_grad()
        img_emb = cnn(img)
        preds = model(ts, img_emb).squeeze()
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
