import numpy as np

def monte_carlo_simulation(model, ts, img, runs=100):
    preds = []
    for _ in range(runs):
        noise = np.random.normal(0, 0.01, ts.shape)
        preds.append(model(ts + noise, img).item())
    return {
        "mean": np.mean(preds),
        "p05": np.percentile(preds, 5),
        "p95": np.percentile(preds, 95),
    }
