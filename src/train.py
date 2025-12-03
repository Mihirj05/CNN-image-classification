import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf

from model import build_cnn
from data_loader import load_cifar10, create_generators, CIFAR10_LABELS
from visualize import plot_history

OUTPUT_DIR = 'outputs'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_augmentation', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--save_best_only', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(normalize=True, val_split=0.1)

    strategy = tf.distribute.get_strategy()
    print("Num devices: ", strategy.num_replicas_in_sync)

    train_gen, val_gen = create_generators(x_train, y_train, x_val, y_val, batch_size=args.batch_size, augment=args.use_augmentation)

    with strategy.scope():
        model = build_cnn(input_shape=x_train.shape[1:], num_classes=10, dropout_rate=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Callbacks
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(MODEL_DIR, f"best_model_{now}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy',
                                           save_best_only=args.save_best_only, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
    ]

    steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
    validation_steps = int(np.ceil(len(x_val) / args.batch_size))

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, f"final_model_{now}.h5")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Plot training history
    plot_history(history, save_path=os.path.join(FIG_DIR, f"training_history_{now}.png"))

if __name__ == '__main__':
    main()
