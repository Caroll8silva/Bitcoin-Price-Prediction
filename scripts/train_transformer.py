import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

TRAIN_X_PATH = 'train_X_tf.npy'
TRAIN_Y_PATH = 'train_y_tf.npy'
MODEL_V2_PATH = 'models/transformer_model.keras'
HEAD_SIZE = 256
NUM_HEADS = 4
FF_DIM = 4
NUM_TRANSFORMER_BLOCKS = 4
MLP_UNITS = [128]
DROPOUT = 0.25
MLP_DROPOUT = 0.4
EPOCHS = 100
BATCH_SIZE = 32 
VALIDATION_SPLIT = 0.2

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.x[batch_indices], self.y[batch_indices]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(inputs.shape[-1])])
    x = ffn(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    return x

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

def main():
    print("--- Training Transformer Model (v2) with Data Generator ---")

    print("ℹ️ Loading pre-processed sequence data (memory-mapped)...")
    X_train_full = np.load(TRAIN_X_PATH, mmap_mode='r')
    y_train_full = np.load(TRAIN_Y_PATH, mmap_mode='r')
    
    split_index = int(len(X_train_full) * (1 - VALIDATION_SPLIT))
    train_indices = np.arange(split_index)
    val_indices = np.arange(split_index, len(X_train_full))
    
    training_generator = DataGenerator(X_train_full[train_indices], y_train_full[train_indices], BATCH_SIZE)
    validation_generator = DataGenerator(X_train_full[val_indices], y_train_full[val_indices], BATCH_SIZE)
    
    print(f"Training with {len(training_generator)} batches per epoch.")
    print(f"Validating with {len(validation_generator)} batches per epoch.")

    input_shape = X_train_full.shape[1:]
    model = build_transformer_model(input_shape, HEAD_SIZE, NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS, MLP_UNITS, DROPOUT, MLP_DROPOUT)

    model.compile(loss="mean_absolute_error", optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=["mean_absolute_error"])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_V2_PATH, save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    print("\nℹ️ Starting model training using data generators...")
    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    print(f"✅ Training complete. Best model saved to {MODEL_V2_PATH}")

if __name__ == "__main__":
    main()