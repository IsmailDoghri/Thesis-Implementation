import numpy as np
import time
import json
import os
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from scipy.spatial.distance import squareform
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

import contextlib
import io


@contextlib.contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


DTAIDISTANCE_AVAILABLE = False
try:
    with suppress_output():
        from dtaidistance import dtw
        from dtaidistance import dtw_ndim
    DTAIDISTANCE_AVAILABLE = True
    print("✓ dtaidistance library available")
except ImportError:
    print("⚠️ dtaidistance not available - using custom implementation")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'results' / 'phase4'
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITIES = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}


class DTWkNNFull:
    def __init__(self, k=5, window=10, use_pruning=True, checkpoint_interval=1000):
        self.k = k
        self.window = window
        self.use_pruning = use_pruning
        self.checkpoint_interval = checkpoint_interval
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.distance_matrix = None
        self.predictions = None
        self.prediction_confidences = None
        self.nearest_neighbors = None
        self.train_time = None
        self.inference_time = None

    def load_data(self):
        print("PHASE 4: DTW-KNN")
        print("Loading data...")

        self.X_train = np.load(DATA_DIR / 'dtw_train_accel_mag.npy')
        self.X_test = np.load(DATA_DIR / 'dtw_test_accel_mag.npy')

        self.y_train = np.load(DATA_DIR / 'y_train.npy')
        self.y_test = np.load(DATA_DIR / 'y_test.npy')

        print(f"✓ Loaded: {self.X_train.shape[0]:,} train, {self.X_test.shape[0]:,} test samples")

        n_train = len(self.X_train)
        n_comparisons = (n_train * (n_train - 1)) // 2
        est_time_per_comp = 0.001 if DTAIDISTANCE_AVAILABLE else 0.0013
        est_total_time = n_comparisons * est_time_per_comp

        print(f"  Estimated time: {est_total_time / 3600:.1f} hours")

    def compute_dtw_distance(self, x, y):
        if DTAIDISTANCE_AVAILABLE:
            with suppress_output():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = dtw.distance(x, y, window=self.window, use_c=True)
            return result
        else:
            return self.dtw_distance_custom(x, y)

    def dtw_distance_custom(self, x, y):
        n, m = len(x), len(y)

        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(max(1, i - self.window), min(m + 1, i + self.window + 1)):
                cost[i, j] = abs(x[i - 1] - y[j - 1]) + min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1]
                )

        return cost[n, m]

    def lb_keogh(self, x, y, r=5):
        n = len(x)
        lower_bound = 0

        for i in range(n):
            lower = min(y[max(0, i - r):min(n, i + r + 1)])
            upper = max(y[max(0, i - r):min(n, i + r + 1)])

            if x[i] > upper:
                lower_bound += (x[i] - upper) ** 2
            elif x[i] < lower:
                lower_bound += (lower - x[i]) ** 2

        return np.sqrt(lower_bound)

    def save_checkpoint(self, distances, idx, start_time):
        checkpoint_path = CHECKPOINT_DIR / 'distance_matrix_checkpoint.pkl'
        checkpoint_data = {
            'distances': distances,
            'last_idx': idx,
            'elapsed_time': time.time() - start_time,
            'n_samples': len(self.X_train)
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"💾 Checkpoint saved at {idx:,}")

    def load_checkpoint(self):
        checkpoint_path = CHECKPOINT_DIR / 'distance_matrix_checkpoint.pkl'
        if checkpoint_path.exists():
            print("📂 Found checkpoint, loading...")
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            return checkpoint_data
        return None

    def compute_distance_matrix_full(self):
        print("Computing DTW distance matrix...")

        n_samples = len(self.X_train)
        n_comparisons = (n_samples * (n_samples - 1)) // 2

        checkpoint = self.load_checkpoint()
        if checkpoint and checkpoint['n_samples'] == n_samples:
            distances = checkpoint['distances']
            start_idx = checkpoint['last_idx'] + 1
            elapsed_time = checkpoint['elapsed_time']
            print(f"  Resuming from {start_idx:,}/{n_comparisons:,}")
            resume = True
        else:
            distances = np.zeros(n_comparisons)
            start_idx = 0
            elapsed_time = 0
            resume = False

        start_time = time.time() - (elapsed_time if resume else 0)
        last_checkpoint_time = time.time()

        if DTAIDISTANCE_AVAILABLE and not resume:
            try:
                print("  Attempting batch computation...")
                with suppress_output():
                    distance_matrix = dtw.distance_matrix_fast(
                        self.X_train,
                        window=self.window,
                        compact=True,
                        parallel=True,
                        use_c=True
                    )
                distances = distance_matrix
                print("✓ Batch computation completed")

            except Exception as e:
                print(f"  Batch failed, using pairwise...")

                idx = start_idx
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        if idx >= start_idx:
                            if self.use_pruning:
                                lb = self.lb_keogh(self.X_train[i], self.X_train[j], r=self.window)
                                if lb > 1000:
                                    distances[idx] = lb
                                else:
                                    distances[idx] = self.compute_dtw_distance(
                                        self.X_train[i], self.X_train[j]
                                    )
                            else:
                                distances[idx] = self.compute_dtw_distance(
                                    self.X_train[i], self.X_train[j]
                                )

                            if (idx + 1) % 1000 == 0:
                                current_time = time.time()
                                elapsed = current_time - start_time
                                progress = (idx + 1) / n_comparisons * 100
                                rate = (idx + 1 - start_idx) / elapsed if elapsed > 0 else 0
                                eta = (n_comparisons - idx - 1) / rate if rate > 0 else 0

                                print(f"  {progress:.1f}% | ETA: {eta / 3600:.1f}h")

                            if (idx + 1) % self.checkpoint_interval == 0:
                                self.save_checkpoint(distances, idx, start_time)
                                last_checkpoint_time = current_time

                        idx += 1
        else:
            idx = start_idx
            pair_count = 0

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if pair_count >= start_idx:
                        distances[idx] = self.compute_dtw_distance(
                            self.X_train[i], self.X_train[j]
                        )

                        if (idx + 1) % 1000 == 0:
                            current_time = time.time()
                            elapsed = current_time - start_time
                            progress = (idx + 1) / n_comparisons * 100
                            rate = (idx + 1 - start_idx) / elapsed if elapsed > 0 else 0
                            eta = (n_comparisons - idx - 1) / rate if rate > 0 else 0

                            print(f"  {progress:.1f}% | ETA: {eta / 3600:.1f}h")

                        if (idx + 1) % self.checkpoint_interval == 0:
                            self.save_checkpoint(distances, idx, start_time)

                        idx += 1
                    pair_count += 1

        self.train_time = time.time() - start_time
        print(f"✓ Distance matrix computed in {self.train_time / 3600:.2f}h")

        self.distance_matrix = squareform(distances)

        matrix_path = RESULTS_DIR / 'distance_matrix_full.npy'
        np.save(matrix_path, self.distance_matrix)
        print(f"✓ Saved distance matrix ({self.distance_matrix.nbytes / (1024 ** 3):.2f} GB)")

        checkpoint_path = CHECKPOINT_DIR / 'distance_matrix_checkpoint.pkl'
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def load_precomputed_matrix(self):
        matrix_path = RESULTS_DIR / 'distance_matrix_full.npy'
        if matrix_path.exists():
            print("Loading precomputed matrix...")
            self.distance_matrix = np.load(matrix_path)
            print(f"✓ Loaded matrix: {self.distance_matrix.shape}")
            return True
        return False

    def predict_full(self):
        print("Making predictions...")

        if self.distance_matrix is None:
            print("❌ Distance matrix not computed!")
            return

        n_test = len(self.X_test)
        self.predictions = np.zeros(n_test, dtype=int)
        self.prediction_confidences = np.zeros(n_test)
        self.nearest_neighbors = []

        start_time = time.time()

        for i, test_sample in enumerate(self.X_test):
            distances_to_train = np.zeros(len(self.X_train))

            for j, train_sample in enumerate(self.X_train):
                distances_to_train[j] = self.compute_dtw_distance(test_sample, train_sample)

            k_nearest_indices = np.argsort(distances_to_train)[:self.k]
            k_nearest_distances = distances_to_train[k_nearest_indices]
            k_nearest_labels = self.y_train[k_nearest_indices]

            weights = 1.0 / (k_nearest_distances + 1e-10)
            weighted_votes = {}

            for label, weight in zip(k_nearest_labels, weights):
                if label not in weighted_votes:
                    weighted_votes[label] = 0
                weighted_votes[label] += weight

            prediction = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
            confidence = weighted_votes[prediction] / sum(weighted_votes.values())

            self.predictions[i] = prediction
            self.prediction_confidences[i] = confidence

            if i < 100:
                self.nearest_neighbors.append({
                    'indices': k_nearest_indices.tolist(),
                    'distances': k_nearest_distances.tolist(),
                    'labels': k_nearest_labels.tolist(),
                    'prediction': int(prediction),
                    'confidence': float(confidence)
                })

            if (i + 1) % 100 == 0 or i == n_test - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / n_test * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_test - i - 1) / rate if rate > 0 else 0

                print(f"  {progress:.1f}% | ETA: {eta / 60:.1f}m")

        self.inference_time = time.time() - start_time
        print(f"✓ Predictions completed in {self.inference_time / 60:.1f}m")

        np.save(RESULTS_DIR / 'predictions_full.npy', self.predictions)
        np.save(RESULTS_DIR / 'prediction_confidences_full.npy', self.prediction_confidences)

    def evaluate(self):
        print("Evaluating...")

        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions, average='weighted')
        recall = recall_score(self.y_test, self.predictions, average='weighted')
        f1 = f1_score(self.y_test, self.predictions, average='weighted')

        print(f"✓ Accuracy: {accuracy:.4f}")

        avg_confidence = np.mean(self.prediction_confidences)

        self.confusion_matrix = confusion_matrix(self.y_test, self.predictions)

        for activity_id in sorted(np.unique(self.y_test)):
            activity_name = ACTIVITIES[activity_id]
            mask = self.y_test == activity_id
            pred_mask = self.predictions == activity_id

            tp = np.sum((self.y_test == activity_id) & (self.predictions == activity_id))
            fp = np.sum((self.y_test != activity_id) & (self.predictions == activity_id))
            fn = np.sum((self.y_test == activity_id) & (self.predictions != activity_id))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_class = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            support = np.sum(mask)

        self.results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_confidence': float(avg_confidence),
            'train_time': float(self.train_time) if self.train_time else None,
            'inference_time': float(self.inference_time),
            'inference_per_sample': float(self.inference_time / len(self.predictions)),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.predictions),
            'k': self.k,
            'window': self.window
        }

        with open(RESULTS_DIR / 'phase4_results_full.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        np.save(RESULTS_DIR / 'confusion_matrix_full.npy', self.confusion_matrix)

        print(f"✓ Results saved to {RESULTS_DIR}")

        return self.results

    def generate_summary(self):
        summary = f"""
PHASE 4: DTW-KNN SUMMARY
================================================================================
Accuracy: {self.results['accuracy']:.4f}
Distance matrix: {(self.train_time / 3600 if self.train_time else 0):.2f}h
Inference: {self.results['inference_time'] / 60:.1f}m
Per sample: {self.results['inference_per_sample']:.3f}s
================================================================================
"""

        with open(RESULTS_DIR / 'phase4_summary_full.txt', 'w') as f:
            f.write(summary)

        print(summary)

        return summary


def main():
    print("PHASE 4: DTW-KNN (FULL DATASET)")
    print("⚠️ This will take ~7-8 hours")

    response = input("Continue? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("Exiting...")
        return

    dtw_model = DTWkNNFull(k=5, window=10, use_pruning=True, checkpoint_interval=10000)

    dtw_model.load_data()

    if not dtw_model.load_precomputed_matrix():
        dtw_model.compute_distance_matrix_full()
    else:
        print("✓ Using precomputed matrix")

    dtw_model.predict_full()

    results = dtw_model.evaluate()

    dtw_model.generate_summary()

    print("✓ PHASE 4 COMPLETED")

    return dtw_model, results


if __name__ == '__main__':
    dtw_model, results = main()
