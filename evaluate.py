"""
ORBITIQ Sentinel — Evaluation, Scoring & Explainability
"""
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, average_precision_score
)

SUBSYSTEMS = {
    "power":     ["bus_voltage","battery_soc","solar_current","power_consumption"],
    "thermal":   ["panel_temp","battery_temp","cpu_temp","radiator_temp"],
    "attitude":  ["rw_speed_x","rw_speed_y","rw_speed_z","pointing_error"],
    "propulsion":["tank_pressure","fuel_remaining","thruster_temp"],
    "comms":     ["signal_strength","bit_error_rate","link_margin"],
}

def get_subsystem(col_name):
    for sub, params in SUBSYSTEMS.items():
        for p in params:
            if col_name == f"{sub}_{p}":
                return sub
    return "unknown"

def anomaly_severity_score(error, threshold, max_multiplier=10):
    """Map reconstruction error to 0-100 severity score"""
    ratio = error / (threshold + 1e-9)
    score = np.clip((ratio - 1.0) / (max_multiplier - 1.0), 0, 1) * 100
    return np.round(score, 1)

def evaluate(model, X_test, labels, threshold, device, col_names):
    """Full evaluation with metrics + per-sample explainability"""
    model.eval()
    tensor = torch.FloatTensor(X_test).to(device)
    errors = model.reconstruction_error(tensor)
    feat_errors = model.feature_errors(tensor)

    preds = (errors > threshold).astype(int)
    scores = anomaly_severity_score(errors, threshold)

    # Metrics
    auc   = roc_auc_score(labels, errors)
    ap    = average_precision_score(labels, errors)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    tn,fp,fn,tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    metrics = {
        "roc_auc": round(float(auc), 4),
        "avg_precision": round(float(ap), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    # Per-sample explainability: top 3 contributing features
    explanations = []
    for i in range(len(X_test)):
        fe = feat_errors[i]
        top3_idx = np.argsort(fe)[::-1][:3]
        top3 = [
            {
                "feature": col_names[j],
                "subsystem": get_subsystem(col_names[j]),
                "contribution": round(float(fe[j] / (fe.sum() + 1e-9) * 100), 1)
            }
            for j in top3_idx
        ]
        explanations.append(top3)

    # Subsystem health scores (mean feature error per subsystem, inverted)
    subsystem_health = {}
    for sub, params in SUBSYSTEMS.items():
        cols = [f"{sub}_{p}" for p in params]
        idxs = [col_names.index(c) for c in cols if c in col_names]
        if idxs:
            mean_err = float(np.mean(feat_errors[:, idxs]))
            # Normalize to 0-100 health (higher = healthier)
            health = max(0, min(100, 100 - mean_err * 2000))
            subsystem_health[sub] = round(health, 1)

    return {
        "errors": errors.tolist(),
        "scores": scores.tolist(),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
        "metrics": metrics,
        "explanations": explanations,
        "subsystem_health": subsystem_health,
        "threshold": float(threshold),
    }
