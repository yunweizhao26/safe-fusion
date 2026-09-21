"""Optional PyTorch selector models used by architecture experiments."""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch import nn


class ContinuousFTTransformer(nn.Module):
    """Small FT-Transformer over continuous selector features."""

    def __init__(
        self,
        n_features: int,
        token_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, token_dim))
        self.bias = nn.Parameter(torch.empty(n_features, token_dim))
        self.cls = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(len(features), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


class FTTransformerClassifier:
    """Scikit-like binary classifier with deterministic early stopping."""

    def __init__(
        self,
        token_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        patience: int = 6,
        validation_fraction: float = 0.1,
        random_state: int = 1729,
        device: str | None = None,
    ) -> None:
        self.token_dim = token_dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device

    def get_params(self, deep: bool = False) -> dict:
        del deep
        return {
            "token_dim": self.token_dim,
            "heads": self.heads,
            "layers": self.layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "device": self.device,
        }

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "FTTransformerClassifier":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        weights = (
            np.ones(len(labels), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        rng = np.random.default_rng(self.random_state)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        validation_parts = []
        training_parts = []
        for indices in (positive, negative):
            shuffled = rng.permutation(indices)
            n_validation = max(1, int(round(self.validation_fraction * len(shuffled))))
            validation_parts.append(shuffled[:n_validation])
            training_parts.append(shuffled[n_validation:])
        validation = np.concatenate(validation_parts)
        training = np.concatenate(training_parts)

        requested_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested_device)
        if device.type == "cpu":
            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        model = ContinuousFTTransformer(
            features.shape[1], self.token_dim, self.heads, self.layers, self.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        feature_tensor = torch.from_numpy(features).to(device)
        label_tensor = torch.from_numpy(labels).to(device)
        weight_tensor = torch.from_numpy(weights).to(device)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.validation_loss_: list[float] = []
        for epoch in range(self.max_epochs):
            model.train()
            epoch_training = rng.permutation(training)
            for start in range(0, len(training), self.batch_size):
                batch = epoch_training[start:start + self.batch_size]
                batch_tensor = torch.from_numpy(batch).to(device)
                logits = model(feature_tensor[batch_tensor])
                losses = criterion(logits, label_tensor[batch_tensor])
                loss = (
                    losses * weight_tensor[batch_tensor]
                ).sum() / weight_tensor[batch_tensor].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            model.eval()
            loss_sum = 0.0
            weight_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(validation), self.batch_size):
                    batch = validation[start:start + self.batch_size]
                    batch_tensor = torch.from_numpy(batch).to(device)
                    losses = criterion(
                        model(feature_tensor[batch_tensor]), label_tensor[batch_tensor]
                    )
                    loss_sum += float(
                        (losses * weight_tensor[batch_tensor]).sum().cpu()
                    )
                    weight_sum += float(weight_tensor[batch_tensor].sum().cpu())
            validation_loss = loss_sum / weight_sum
            self.validation_loss_.append(validation_loss)
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("FT-Transformer training did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.device_ = device
        self.n_iter_ = epoch + 1
        self.best_validation_loss_ = best_loss
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        probabilities: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                batch = torch.from_numpy(features[start:start + self.batch_size]).to(self.device_)
                probabilities.append(torch.sigmoid(self.model_(batch)).cpu().numpy())
        positive = np.concatenate(probabilities).astype(np.float32, copy=False)
        return np.column_stack([1.0 - positive, positive])
