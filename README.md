# Multimodal Transformer-Based Short-Term PV Power Forecasting

## Overview
This repository implements a multimodal AI framework for short-term photovoltaic (PV) power forecasting by integrating:
- PV electrical measurements
- Weather sensor data
- Sky-image visual information

The system leverages Transformer-based architectures with multimodal fusion to enable proactive and reliable solar power prediction.

## Key Contributions
- Multimodal Transformer for PV forecasting
- CNN / Vision Transformer-based sky image encoding
- Ablation studies for modality contribution
- Probabilistic forecasting with uncertainty estimation
- Digital twin integration for decision intelligence
- Anomaly detection for PV system monitoring

## Dataset
- **Stanford 2019 Sky Images and Photovoltaic Power Generation Dataset**
- Short-term forecasting horizon: minutes to hours ahead

## Repository Structure
- `notebooks/` – Research and experimentation
- `models/` – Neural architectures
- `training/` – Training pipelines
- `digital_twin/` – Physics-informed modeling
- `anomaly_detection/` – Fault detection
- `dashboard/` – Visualization (deployment-ready)

## Results
The multimodal Transformer achieves:
- **22.5% RMSE improvement** over persistence baseline
- Statistically significant gains (Wilcoxon test, p < 0.05)

## Future Work
- Vision Transformer + Graph Neural Network integration
- Real-time deployment using edge–cloud architecture
- Regional multi-site forecasting
- Probabilistic and risk-aware grid decision support

## Author
**Vivek Palsutkar**  
Proposed Supervisor: **Dr. Yang Hu**  
Beihang University
