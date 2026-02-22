from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class ModelConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_layers: List[int] = None
    dropout: float = 0.15
    early_stopping_patience: int = 5


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_ann(input_dim: int, hidden_layers: List[int], dropout: float) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for h in hidden_layers:
        x = layers.Dense(h, activation="relu")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs)


def train_ann_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ModelConfig,
) -> Tuple[keras.Model, Dict[str, Any]]:
    hidden = cfg.hidden_layers or [128, 64, 32]
    model = build_ann(X_train.shape[1], hidden, cfg.dropout)

    opt = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    return model, {"history": history.history, "hidden_layers": hidden}
