# CIFAR-10 CNN Ablation Study

A controlled TensorFlow/Keras study of training duration, CNN architecture, augmentation,
dropout, and learning-rate scheduling. The selected CNN was trained from scratch and
achieved **87.51% mean accuracy** on the official CIFAR-10 test set across three fixed
seeds, exceeding the 78% target by 9.51 percentage points.

## Results

| Run | Mean validation accuracy ± SD | Mean test accuracy ± SD |
| --- | ---: | ---: |
| Original CNN, two epochs | 38.15% ± 8.85 pp | 38.00% ± 9.02 pp |
| Original CNN, 30 epochs | 65.40% ± 1.02 pp | |
| **Selected 30-epoch CNN** | **87.92% ± 0.20 pp** | **87.51% ± 0.08 pp** |
| Restore original architecture | 76.66% ± 0.50 pp | |
| Restore original augmentation | 84.39% ± 0.34 pp | |
| Restore 0.50 dropout | 87.53% ± 0.47 pp | |
| Restore constant learning rate | 85.91% ± 0.62 pp | |

The selected CNN scored **87.45%, 87.60%, and 87.48%** on the test set for
seeds 17, 29, and 43. The reproduced baseline scored 27.63%, 44.02%, and 42.35%.
Standard deviations are sample estimates over the three seeds.

- Within the original CNN's 30-epoch runs, the best mean validation accuracy
  rose from 44.38% in the first two epochs to 65.40% over all 30 epochs.
- Restoring the original architecture produced the largest validation drop
  among the four reversions (11.26 percentage points).
- Restoring the original augmentation or constant learning rate lowered
  mean validation accuracy by 3.53 and 2.01 points, respectively. Restoring
  0.50 dropout lowered it by 0.39 points, with the direction varying by seed.
- Dog images classified as cats were the most frequent error for each seed
  of the selected CNN.

The [run data](results/controlled-study.json) includes all 21 training histories,
per-seed metrics, checkpoint epochs, and the six test confusion matrices.

## Evaluation protocol

The official training set is split into 45,000 training images and 5,000 validation
images, with 500 validation images per class. All runs use the same split (seed 2026),
training seeds (17, 29, 43), batch size (64), Adam optimizer, and starting learning rate
(0.001). Each run saves its highest-validation-accuracy checkpoint, using the first
checkpoint in a tie. The highest mean validation accuracy among the 30-epoch
configurations determines which one to test alongside the reproduced baseline.

The [original notebook](notebooks/cifar10_experiments.ipynb) is historical. Its
45.40% baseline was logged after two epochs with the official test set passed as
validation data. The new baseline preserves the original architecture, training
duration, and augmentation, and uses the new split and checkpoint rule.

## Experiment matrix

| Run | Comparison | Changed factor |
| --- | --- | --- |
| `baseline` | Historical reference under the new split | Original two-epoch recipe |
| `baseline_30` | First two epochs of the same run | 30 epochs |
| `final` | `baseline` | Combined 30-epoch candidate |
| `original_architecture` | `final` | Original three-convolution architecture |
| `original_augmentation` | `final` | Rotation and zoom in place of translation |
| `original_dropout` | `final` | Dropout 0.5 instead of 0.25 |
| `constant_lr` | `final` | Constant learning rate in place of cosine decay |

The candidate uses two 3×3 convolutions per block, batch normalization after each
convolution, and global average pooling before the classifier. Its augmentation is
horizontal flip and reflected translation up to four pixels. The architecture
comparison measures that bundle of changes together. The runner applies the
original flip, rotation, and zoom distributions with batched TensorFlow operations.

## Reproduce

Use Python 3.9. For native Windows GPU training, TensorFlow 2.10.1 requires CUDA
11.2 and cuDNN 8.1. The commands below use mixed precision on a compatible GPU.
CIFAR-10 is downloaded on the first run.

```sh
python -m pip install -r requirements.txt
python study.py train --mixed-precision --experiments baseline baseline_30 final original_architecture original_augmentation original_dropout constant_lr
python study.py test --mixed-precision --experiments baseline final
```

Each run writes epoch metrics, split fingerprint, run configuration, and checkpoint
selection to `runs/<experiment>/<seed>/validation.json`. Test evaluation writes
accuracy and a 10-class confusion matrix to `test.json` in the same directory. See the
[experiment protocol](docs/Experiment%20specifications.txt) for comparison details.

## Historical materials

The [original report](results/experiment-summary.pdf) and its plots describe the
original notebook runs and their evaluation protocol.
