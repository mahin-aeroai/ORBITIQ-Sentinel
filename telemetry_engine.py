"""
ORBITIQ Sentinel — Layer 5: Telemetry Anomaly Detection Engine
Generates synthetic spacecraft telemetry + trains LSTM Autoencoder
"""
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

np.random.seed(42)

SUBSYSTEMS = {
    "power": {
        "bus_voltage":       {"normal": (28.0, 0.3),  "unit": "V",   "anomaly": "spike"},
        "battery_soc":       {"normal": (82.0, 3.0),  "unit": "%",   "anomaly": "drift_down"},
        "solar_current":     {"normal": (4.2,  0.2),  "unit": "A",   "anomaly": "dropout"},
        "power_consumption": {"normal": (120.0, 5.0), "unit": "W",   "anomaly": "spike"},
    },
    "thermal": {
        "panel_temp":        {"normal": (22.0, 4.0),  "unit": "°C",  "anomaly": "drift_up"},
        "battery_temp":      {"normal": (18.0, 2.0),  "unit": "°C",  "anomaly": "drift_up"},
        "cpu_temp":          {"normal": (45.0, 3.0),  "unit": "°C",  "anomaly": "spike"},
        "radiator_temp":     {"normal": (-10.0, 5.0), "unit": "°C",  "anomaly": "drift_up"},
    },
    "attitude": {
        "rw_speed_x":        {"normal": (1200, 80),   "unit": "rpm", "anomaly": "oscillation"},
        "rw_speed_y":        {"normal": (1150, 80),   "unit": "rpm", "anomaly": "oscillation"},
        "rw_speed_z":        {"normal": (1100, 80),   "unit": "rpm", "anomaly": "oscillation"},
        "pointing_error":    {"normal": (0.05, 0.02), "unit": "deg", "anomaly": "drift_up"},
    },
    "propulsion": {
        "tank_pressure":     {"normal": (220.0, 2.0), "unit": "bar", "anomaly": "drift_down"},
        "fuel_remaining":    {"normal": (68.0, 0.1),  "unit": "%",   "anomaly": "drift_down"},
        "thruster_temp":     {"normal": (35.0, 3.0),  "unit": "°C",  "anomaly": "spike"},
    },
    "comms": {
        "signal_strength":   {"normal": (-85.0, 3.0), "unit": "dBm", "anomaly": "drift_down"},
        "bit_error_rate":    {"normal": (0.001, 0.0003), "unit": "",  "anomaly": "spike"},
        "link_margin":       {"normal": (12.0, 1.0),  "unit": "dB",  "anomaly": "drift_down"},
    },
}

PARAMS = []
for sub, params in SUBSYSTEMS.items():
    for name in params:
        PARAMS.append((sub, name))

N_PARAMS = sum(len(v) for v in SUBSYSTEMS.values())

def generate_normal(n_steps=1440):
    """Generate 1440 steps (24h @ 1-min cadence) of normal telemetry"""
    t = np.linspace(0, 4*np.pi, n_steps)
    data = {}
    for sub, params in SUBSYSTEMS.items():
        for name, cfg in params.items():
            mu, sigma = cfg["normal"]
            # add sinusoidal orbital variation + noise
            orbital = 0.3 * sigma * np.sin(t + np.random.uniform(0, 2*np.pi))
            noise = np.random.normal(0, sigma, n_steps)
            data[f"{sub}_{name}"] = mu + orbital + noise
    return pd.DataFrame(data)

def inject_anomaly(df, col, atype, start, duration=60):
    """Inject a specific anomaly type into a column"""
    df = df.copy()
    end = min(start + duration, len(df))
    idx = slice(start, end)
    vals = df[col].values.copy()
    sub = col.split("_")[0]
    param = "_".join(col.split("_")[1:])
    mu, sigma = SUBSYSTEMS[sub][param]["normal"]

    if atype == "spike":
        vals[start:end] += np.random.uniform(6, 10) * sigma * np.sign(np.random.randn())
    elif atype == "drift_up":
        vals[start:end] += np.linspace(0, 8 * sigma, end - start)
    elif atype == "drift_down":
        vals[start:end] -= np.linspace(0, 8 * sigma, end - start)
    elif atype == "dropout":
        vals[start:end] = mu * 0.05
    elif atype == "oscillation":
        vals[start:end] += 5 * sigma * np.sin(np.linspace(0, 10*np.pi, end - start))

    df[col] = vals
    return df

def generate_dataset(n_normal=2000, n_anomaly=400, seq_len=60):
    """Build train/test sequences with labels"""
    print("Generating telemetry dataset...")
    all_data = generate_normal(n_steps=n_normal + n_anomaly + seq_len + 200)

    # Inject 8 random anomalies
    injected = []
    for _ in range(8):
        sub = np.random.choice(list(SUBSYSTEMS.keys()))
        param = np.random.choice(list(SUBSYSTEMS[sub].keys()))
        col = f"{sub}_{param}"
        atype = SUBSYSTEMS[sub][param]["anomaly"]
        start = np.random.randint(n_normal, n_normal + n_anomaly - 120)
        dur = np.random.randint(40, 90)
        all_data = inject_anomaly(all_data, col, atype, start, dur)
        injected.append({"col": col, "start": start, "end": start+dur, "type": atype})

    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(all_data.values)

    # Build sequences
    def make_seqs(arr, start, n):
        seqs = []
        for i in range(start, start + n - seq_len):
            seqs.append(arr[i:i+seq_len])
        return np.array(seqs, dtype=np.float32)

    X_train = make_seqs(scaled, 0, n_normal)
    X_test  = make_seqs(scaled, n_normal, n_anomaly)

    # Labels: 1 if any timestep in window overlaps injected anomaly
    labels = np.zeros(len(X_test))
    for inj in injected:
        for i in range(len(X_test)):
            win_start = n_normal + i
            win_end   = win_start + seq_len
            if win_start < inj["end"] and win_end > inj["start"]:
                labels[i] = 1

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Anomalies injected: {len(injected)}")
    print(f"  Anomaly labels (test): {int(labels.sum())} / {len(labels)}")
    return X_train, X_test, labels, scaler, all_data, injected, list(all_data.columns)

if __name__ == "__main__":
    X_train, X_test, labels, scaler, df, injected, cols = generate_dataset()
    print("Columns:", cols[:5], "...")
    print("Done.")
