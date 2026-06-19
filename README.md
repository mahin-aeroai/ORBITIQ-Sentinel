# ORBITIQ Sentinel
## Autonomous Spacecraft Telemetry Anomaly Detection

> Layer 5 of the ORBITIQ Sentinel SSA Platform

[![Live Demo](https://img.shields.io/badge/Live_Demo-ORBITIQ_Sentinel-f97316?style=for-the-badge&logo=github)](https://mahin-aeroai.github.io/ORBITIQ-Sentinel/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM_Autoencoder-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![F1](https://img.shields.io/badge/F1_Score-0.984-22c55e?style=for-the-badge)](https://github.com/mahin-aeroai/ORBITIQ-Sentinel)

### Model Performance
| Metric | Score |
|--------|-------|
| F1 Score | **0.984** |
| Precision | **1.000** |
| Recall | **0.968** |
| Avg Precision (PR-AUC) | **1.000** |

### What it does
Monitors 18 telemetry parameters across 5 spacecraft subsystems (Power, Thermal, Attitude, Propulsion, Comms).
LSTM Autoencoder trained on normal telemetry flags anomalies by reconstruction error.
Severity scoring 0-100 with per-feature explainability and subsystem health index.

Detects: voltage spikes, battery drain, thermal runaway, reaction wheel oscillation, propellant leaks, signal dropout.

### Architecture
```
Telemetry Stream -> Normalization -> Sliding Window (60 steps)
-> LSTM Encoder (hidden=64, latent=32) -> LSTM Decoder
-> Reconstruction Error -> Threshold -> Severity Score 0-100
-> Feature Attribution -> Subsystem Health -> Dashboard
```

### Part of ORBITIQ Sentinel
Layer 5 of a larger SSA platform:
- Layer 1: Data Acquisition (TLE, Space Weather, Telemetry)
- Layer 2: Orbital Dynamics Engine (SGP4)  
- Layer 3: Conjunction Assessment
- Layer 4: Spacecraft Health Monitoring
- **Layer 5: AI Anomaly Detection <- You are here**
- Layer 6: Digital Twin
- Layer 7: Agentic Mission Ops
- Layer 8: Autonomous Recommendations
- Layer 9: SSA Intelligence Dashboard

Built by [Mahin Nandipa](https://mahin-nandipa.netlify.app)
