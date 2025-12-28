import torch

def run_inference(model, cnn, ts_data, img_data):
    model.eval()
    cnn.eval()
    with torch.no_grad():
        img_emb = cnn(img_data)
        forecast = model(ts_data, img_emb)
    return forecast
