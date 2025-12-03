import argparse
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import build_cnn
from data_loader import load_cifar10, CIFAR10_LABELS
from visualize import plot_confusion_matrix, print_classification_report

OUTPUT_DIR = 'outputs'
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_cifar', action='store_true', default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(normalize=True)
    print("Loaded test data:", x_test.shape, y_test.shape)

   
    model = tf.keras.models.load_model(args.model_path)
    print("Model loaded from", args.model_path)

   
    y_pred_probs = model.predict(x_test, batch_size=64)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    plot_confusion_matrix(y_test, y_pred, classes=CIFAR10_LABELS, normalize=False, save_path=os.path.join(FIG_DIR, 'confusion_matrix.png'))
    print_classification_report(y_test, y_pred, CIFAR10_LABELS)

if __name__ == '__main__':
    main()
