#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import json

np.random.seed(42)
tf.random.set_seed(42)


class CNNClassifier:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.activities = {
            0: 'WALKING',
            1: 'WALKING_UPSTAIRS',
            2: 'WALKING_DOWNSTAIRS',
            3: 'SITTING',
            4: 'STANDING',
            5: 'LAYING'
        }

    def build_model(self, input_shape, n_classes=6):
        model = models.Sequential([
            layers.Input(shape=input_shape),

            layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.1),

            layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

            layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def load_raw_signals(self):
        print("Loading raw signals...")
        data_path = Path("../UCI HAR Dataset")

        train_signals = []
        test_signals = []

        signal_names = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        for signal_name in signal_names:
            train_file = data_path / "train" / "Inertial Signals" / f"{signal_name}_train.txt"
            test_file = data_path / "test" / "Inertial Signals" / f"{signal_name}_test.txt"
            train_signals.append(np.loadtxt(train_file))
            test_signals.append(np.loadtxt(test_file))

        X_train = np.stack(train_signals, axis=-1)
        X_test = np.stack(test_signals, axis=-1)

        y_train = np.loadtxt(data_path / "train" / "y_train.txt", dtype=int) - 1
        y_test = np.loadtxt(data_path / "test" / "y_test.txt", dtype=int) - 1

        print(f"✓ Loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

        return X_train, X_test, y_train, y_test

    def normalize_data(self, X_train, X_test):
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_timesteps = X_train.shape[1]
        n_channels = X_train.shape[2]

        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)

        X_train_normalized = self.scaler.fit_transform(X_train_flat)
        X_test_normalized = self.scaler.transform(X_test_flat)

        X_train_normalized = X_train_normalized.reshape(n_train, n_timesteps, n_channels)
        X_test_normalized = X_test_normalized.reshape(n_test, n_timesteps, n_channels)

        return X_train_normalized, X_test_normalized

    def train(self, X_train, y_train, X_test, y_test):
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_model(input_shape)

        total_params = self.model.count_params()
        print(f"Model parameters: {total_params:,}")

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        print("Training CNN...")
        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        training_time = time.time() - start_time

        print(f"✓ Training completed in {training_time:.2f}s")
        print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")

        return history, training_time

    def evaluate(self, X_test, y_test):
        start_time = time.time()
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)

        print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        target_names = [self.activities[i] for i in range(6)]
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        for i in range(6):
            mask = y_test == i
            if np.sum(mask) > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])

        return {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'predictions': y_pred.tolist(),
            'confusion_matrix': cm.tolist(),
            'training_time': getattr(self, 'training_time', None)
        }


def main():
    print("PHASE 5: CNN IMPLEMENTATION")

    cnn = CNNClassifier()

    X_train, X_test, y_train, y_test = cnn.load_raw_signals()

    X_train_norm, X_test_norm = cnn.normalize_data(X_train, X_test)

    history, training_time = cnn.train(X_train_norm, y_train, X_test_norm, y_test)
    cnn.training_time = training_time

    results = cnn.evaluate(X_test_norm, y_test)
    cnn.test_accuracy = results['accuracy']
    results['training_time'] = training_time

    with open('cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    cnn.model.save('har_cnn_model.keras')
    print(f"✓ Model saved: har_cnn_model.keras")
    print(f"✓ Results saved: cnn_results.json")

    print("✓ PHASE 5 COMPLETED")

    return results


if __name__ == "__main__":
    results = main()
