"""Train and evaluate the controlled CIFAR-10 study."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class Config:
    epochs: int
    architecture: str
    augmentation: str
    dropout: float
    schedule: str


BASELINE = Config(2, "original", "original", 0.5, "constant")
FINAL = Config(30, "conv_blocks", "translate_flip", 0.25, "cosine")
CONFIGS = {
    "baseline": BASELINE,
    "baseline_30": replace(BASELINE, epochs=FINAL.epochs),
    "final": FINAL,
    "original_architecture": replace(FINAL, architecture=BASELINE.architecture),
    "original_augmentation": replace(FINAL, augmentation=BASELINE.augmentation),
    "original_dropout": replace(FINAL, dropout=BASELINE.dropout),
    "constant_lr": replace(FINAL, schedule=BASELINE.schedule),
}
SEEDS = (17, 29, 43)
BATCH_SIZE = 64
SPLIT_SEED = 2026
LEARNING_RATE = 0.001


def split_indices(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    by_class = [rng.permutation(np.flatnonzero(labels == label)) for label in range(10)]
    validation = np.sort(np.concatenate([indices[:500] for indices in by_class]))
    train = np.sort(np.concatenate([indices[500:] for indices in by_class]))
    return train, validation


def batches(images, labels, seed=None):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if seed is not None:
        dataset = dataset.shuffle(len(images), seed=seed)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class Augmentation(layers.Layer):
    """Apply independent per-image transforms in batched TensorFlow ops."""

    def __init__(self, recipe, seed):
        super().__init__(dtype="float32")
        self.recipe = recipe
        self.random = tf.random.Generator.from_seed(seed)

    @staticmethod
    def transform(images, a, b, x, c, d, y):
        zero = tf.zeros_like(a)
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images, transforms=tf.stack((a, b, x, c, d, y, zero, zero), axis=1),
            output_shape=(32, 32), interpolation="BILINEAR", fill_mode="REFLECT",
            fill_value=0.0,
        )

    def call(self, images, training=False):
        if not training:
            return images
        shape = (tf.shape(images)[0],)
        flip = self.random.uniform(shape) < 0.5
        images = tf.where(flip[:, None, None, None], tf.reverse(images, [2]), images)
        if self.recipe == "original":
            angle = self.random.uniform(shape, -0.2 * np.pi, 0.2 * np.pi)
            cos, sin = tf.cos(angle), tf.sin(angle)
            center = 15.5
            images = self.transform(
                images, cos, -sin, center * (1 - cos + sin),
                sin, cos, center * (1 - cos - sin),
            )
            zoom = self.random.uniform(shape, 0.9, 1.1)
            offset = center * (1 - zoom)
            zero = tf.zeros_like(zoom)
            return self.transform(images, zoom, zero, offset, zero, zoom, offset)
        x = self.random.uniform(shape, -4.0, 4.0)
        y = self.random.uniform(shape, -4.0, 4.0)
        zero, one = tf.zeros_like(x), tf.ones_like(x)
        return self.transform(images, one, zero, x, zero, one, y)


def build_model(config, seed):
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3), dtype="uint8"),
        layers.Rescaling(1 / 255, dtype="float32"),
        Augmentation(config.augmentation, seed),
    ])

    if config.architecture == "original":
        for width in (32, 64, 128):
            model.add(layers.Conv2D(width, 3, activation="relu"))
            model.add(layers.MaxPooling2D())
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        for units in (128, 64):
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(config.dropout))
    else:
        for width in (32, 64, 128):
            for _ in range(2):
                model.add(layers.Conv2D(width, 3, padding="same", use_bias=False))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation("relu"))
            model.add(layers.MaxPooling2D())
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(10, activation="softmax", dtype="float32"))
    return model


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def train(config_name, seed, run, images, labels, train_idx, val_idx):
    config = CONFIGS[config_name]
    run.mkdir(parents=True, exist_ok=True)
    training = batches(images[train_idx], labels[train_idx], seed)
    validation = batches(images[val_idx], labels[val_idx])
    model = build_model(config, seed)
    learning_rate = LEARNING_RATE
    if config.schedule == "cosine":
        learning_rate = keras.optimizers.schedules.CosineDecay(
            LEARNING_RATE, config.epochs * len(training), alpha=0.03
        )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                  steps_per_execution=16)
    start = perf_counter()
    history = model.fit(
        training, validation_data=validation, epochs=config.epochs, verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(
            str(run / "best.weights.h5"), monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=True,
        )],
    ).history
    best = int(np.argmax(history["val_accuracy"]))
    save_json(run / "validation.json", {
        "experiment": config_name,
        "config": asdict(config),
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "validation_indices_sha256": hashlib.sha256(val_idx.astype("<i8").tobytes()).hexdigest(),
        "batch_size": BATCH_SIZE,
        "tensorflow": tf.__version__,
        "precision": keras.mixed_precision.global_policy().name,
        "gpu": [tf.config.experimental.get_device_details(device).get("device_name")
                for device in tf.config.list_physical_devices("GPU")],
        "parameters": model.count_params(),
        "seconds": round(perf_counter() - start, 2),
        "best_epoch": best + 1,
        "validation_accuracy": history["val_accuracy"][best],
        "history": history,
    })


def evaluate(config_name, seed, run, images, labels):
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    if (validation["experiment"] != config_name or validation["seed"] != seed
            or validation["config"] != asdict(CONFIGS[config_name])):
        raise ValueError(f"Checkpoint metadata does not match {config_name} seed {seed}")
    if validation["precision"] != keras.mixed_precision.global_policy().name:
        raise ValueError("Use the same precision policy for training and test evaluation")
    model = build_model(CONFIGS[config_name], seed)
    model.load_weights(str(run / "best.weights.h5"))
    probabilities = model.predict(batches(images, labels), verbose=0)
    predictions = probabilities.argmax(axis=1)
    matrix = tf.math.confusion_matrix(labels, predictions, num_classes=10).numpy()
    save_json(run / "test.json", {
        "experiment": config_name,
        "seed": seed,
        "best_epoch": validation["best_epoch"],
        "test_accuracy": float(np.trace(matrix) / matrix.sum()),
        "confusion_matrix": matrix.tolist(),
    })
    print(f"{config_name} seed {seed}: {np.trace(matrix) / matrix.sum():.2%} test accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--experiments", nargs="+", choices=CONFIGS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--mixed-precision", action="store_true")
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    (images, labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    labels = labels.ravel()
    train_idx, val_idx = split_indices(labels)
    for name in args.experiments:
        for seed in args.seeds:
            keras.backend.clear_session()
            keras.utils.set_random_seed(seed)
            run = args.run_dir / name / str(seed)
            if args.mode == "train":
                if (run / "validation.json").exists():
                    raise FileExistsError(f"{run} already contains a completed run")
                train(name, seed, run, images, labels, train_idx, val_idx)
            else:
                evaluate(name, seed, run, test_images, test_labels.ravel())


if __name__ == "__main__":
    main()
