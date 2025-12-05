"""Sequence-aware business cycle models using LSTM with attention.

This module introduces a temporal model to replace purely row-wise
approaches. With ~185 monthly observations and hundreds of engineered
features (lags, pct_change, composites), treating each month
independently wastes the time-order structure and risks overfitting.

The LSTM + attention architecture learns dependencies across rolling
windows while a mutual-information feature screen caps the dimensionality
for the small sample size.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Model components
# ----------------------------


class AttentionLayer(nn.Module):
    """Additive attention over the time dimension for sequence outputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # outputs: (batch, seq_len, hidden)
        proj = torch.tanh(self.proj(outputs))
        attn_logits = self.score(proj).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(attn_logits, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.lstm(x)
        context, weights = self.attention(outputs)
        logits = self.classifier(self.dropout(context))
        return logits, weights


@dataclass
class SequencePipeline:
    model_state: Dict[str, torch.Tensor]
    label_encoder: LabelEncoder
    feature_order: List[str]
    scaler: StandardScaler
    lookback: int
    feature_scores: Dict[str, float]
    train_accuracy: float
    val_accuracy: Optional[float]


# ----------------------------
# Feature selection & sequence creation
# ----------------------------


def _select_top_features(X: pd.DataFrame, y: pd.Series, max_features: int = 64) -> Tuple[pd.DataFrame, pd.Series]:
    """Reduce dimensionality using mutual information scores."""

    if X.empty:
        return X, pd.Series(dtype=float)

    y_encoded = LabelEncoder().fit_transform(y)
    scores = mutual_info_classif(X, y_encoded, random_state=42, discrete_features=False)
    score_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    selected = score_series.head(max_features).index
    return X[selected], score_series


def _build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Create rolling-window sequences aligned to the target labels."""

    if X.empty or y.dropna().empty:
        return np.empty((0, lookback, 0)), np.empty((0,)), []

    X_filled = X.ffill().bfill()
    y_clean = y.dropna()
    X_aligned = X_filled.reindex(y_clean.index).dropna()
    y_aligned = y_clean.reindex(X_aligned.index)

    sequences: List[np.ndarray] = []
    targets: List = []
    indices: List[pd.Timestamp] = []

    for i in range(lookback - 1, len(X_aligned)):
        window = X_aligned.iloc[i - lookback + 1 : i + 1]
        if window.isna().any().any():
            continue
        sequences.append(window.values)
        targets.append(y_aligned.iloc[i])
        indices.append(window.index[-1])

    return np.asarray(sequences), np.asarray(targets), indices


# ----------------------------
# Training helpers
# ----------------------------


def _train_single_horizon(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = 18,
    hidden_size: int = 64,
    num_layers: int = 1,
    max_features: int = 64,
    max_epochs: int = 60,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 8,
) -> Optional[SequencePipeline]:
    """Train an attention LSTM classifier for one horizon."""

    X_selected, feature_scores = _select_top_features(X, y, max_features=max_features)
    scaler = StandardScaler().fit(X_selected)
    X_scaled = pd.DataFrame(scaler.transform(X_selected), index=X_selected.index, columns=X_selected.columns)

    sequences, targets, _ = _build_sequences(X_scaled, y, lookback)
    if len(sequences) == 0:
        return None

    label_encoder = LabelEncoder().fit(targets)
    targets_enc = label_encoder.transform(targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    X_tensor = torch.tensor(sequences, dtype=torch.float32)
    y_tensor = torch.tensor(targets_enc, dtype=torch.long)

    # time-aware split: last 20% for validation
    split_idx = int(len(X_tensor) * 0.8)
    if split_idx <= 0:
        split_idx = len(X_tensor) - 1
    train_ds = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
    val_ds = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None

    model = LSTMAttentionModel(
        input_dim=X_tensor.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = -math.inf
    epochs_without_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        # Validation
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                    logits, _ = model(val_x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == val_y).sum().item()
                    total += len(val_y)
            val_acc = correct / total if total else None
        else:
            val_acc = None

        if val_acc is None or val_acc > best_val:
            best_val = val_acc if val_acc is not None else 1.0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    # Final train accuracy for logging
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_tensor.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    train_acc = float((preds == targets_enc).mean())

    return SequencePipeline(
        model_state=best_state,
        label_encoder=label_encoder,
        feature_order=list(X_selected.columns),
        scaler=scaler,
        lookback=lookback,
        feature_scores=feature_scores.to_dict(),
        train_accuracy=train_acc,
        val_accuracy=None if val_loader is None else float(best_val),
    )


# ----------------------------
# Public APIs
# ----------------------------


def train_lstm_multi_horizon(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    horizons: Sequence[str] = ("cycle_1m", "cycle_3m", "cycle_6m"),
    model_dir: str = "models",
    lookback: int = 18,
) -> Tuple[Dict[str, Optional[SequencePipeline]], Dict[str, Optional[float]]]:
    """Train LSTM+attention models for each target horizon."""

    os.makedirs(model_dir, exist_ok=True)
    pipelines: Dict[str, Optional[SequencePipeline]] = {}
    accuracies: Dict[str, Optional[float]] = {}

    for horizon in horizons:
        y = df_targets[horizon].reindex(df_features.index).dropna()
        X = df_features.reindex(y.index)
        pipeline = _train_single_horizon(X, y, lookback=lookback)
        pipelines[horizon] = pipeline
        accuracies[horizon] = None if pipeline is None else pipeline.train_accuracy
    return pipelines, accuracies


def _prepare_inference_window(X_all: pd.DataFrame, feature_order: List[str], lookback: int, as_of: Optional[str]) -> pd.DataFrame:
    """Slice the last `lookback` rows ending at as_of (or latest)."""

    X_sorted = X_all.sort_index()
    if as_of is not None:
        ts = pd.to_datetime(as_of)
        ts = ts.to_period("M").to_timestamp("M")
        if ts not in X_sorted.index:
            raise KeyError(f"No feature row found for as_of={as_of}")
        end_loc = X_sorted.index.get_loc(ts)
    else:
        end_loc = len(X_sorted) - 1

    start_loc = end_loc - lookback + 1
    if start_loc < 0:
        raise ValueError("Not enough history to build a sequence window")

    window = X_sorted.iloc[start_loc : end_loc + 1]
    return window.reindex(columns=feature_order)


def predict_with_lstm_attention(
    pipelines: Dict[str, Optional[SequencePipeline]],
    X_all: pd.DataFrame,
    top_n: int = 10,
    as_of: Optional[str] = None,
) -> Dict[str, Dict]:
    """Generate predictions and feature summaries for each horizon."""

    results = {"as_of": (pd.to_datetime(as_of).isoformat() if as_of else X_all.index[-1].isoformat())}

    for horizon, pipeline in pipelines.items():
        if pipeline is None:
            results[horizon] = {
                "predicted_phase": None,
                "probabilities": {},
                "top_features": [],
                "most_impactful_indicators": [],
            }
            continue

        window = _prepare_inference_window(X_all, pipeline.feature_order, pipeline.lookback, as_of)
        scaled = pipeline.scaler.transform(window)
        window_tensor = torch.tensor(np.expand_dims(scaled, axis=0), dtype=torch.float32)

        model = LSTMAttentionModel(
            input_dim=window_tensor.shape[-1],
            hidden_size=64,
            num_layers=1,
            num_classes=len(pipeline.label_encoder.classes_),
            dropout=0.0,
        )
        model.load_state_dict(pipeline.model_state)
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = pipeline.label_encoder.inverse_transform([pred_idx])[0]
        prob_dict = {label: float(prob) for label, prob in zip(pipeline.label_encoder.classes_, probs)}

        # Rank features using mutual-information scores (static) and last attention weights (temporal focus)
        feature_scores = pd.Series(pipeline.feature_scores)
        top_features = (
            feature_scores.reindex(pipeline.feature_order)
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={"index": "feature", 0: "impact"})
            .to_dict(orient="records")
        )

        # Aggregate feature impacts by base indicator
        base_impacts: Dict[str, float] = {}
        for feat, score in feature_scores.items():
            base = feat.split("_lag")[0].replace("_pct", "")
            base_impacts[base] = base_impacts.get(base, 0.0) + abs(float(score))
        top_indicators = [
            {"indicator": k, "aggregate_impact": v}
            for k, v in sorted(base_impacts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        ]

        results[horizon] = {
            "predicted_phase": pred_label,
            "probabilities": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
            "top_features": top_features,
            "most_impactful_indicators": top_indicators,
            "attention_weights": attn_weights.squeeze(0).tolist(),
        }

    return results

