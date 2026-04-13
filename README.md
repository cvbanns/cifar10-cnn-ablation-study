# CIFAR-10 CNN Ablation Study 📊🧠

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Task](https://img.shields.io/badge/Task-Image%20Classification-purple)
![Notebook](https://img.shields.io/badge/Workflow-Jupyter-informational?logo=jupyter)
![Study Type](https://img.shields.io/badge/Study-Ablation%20Analysis-red)

A controlled, experiment-driven study of how common CNN design decisions affect image classification performance on the **CIFAR-10** dataset.

Instead of treating the project like a race for one final score, this repo focuses on something more useful: **what changed, what improved, what got worse, and why**.

---

## ✨ Project Overview

This project explores how standard convolutional neural network choices shape performance on small natural images.

Starting from a baseline CNN, I ran a sequence of controlled experiments around:

- 🎨 Data augmentation
- 🧱 CNN architecture
- 📐 Convolution filter size
- 🏊 Pooling strategy
- 🛡️ Regularization
- ⚙️ Optimizer choice
- 📉 Learning rate
- ⏱️ Epoch count
- 📦 Batch size
- 🔍 Prediction and error analysis

The work follows the structure of the original assignment while packaging it as a cleaner, experiment-based repo that shows the actual behavior of the model across design changes.

---

## 🧪 What This Study Tries to Answer

This repo is built around a simple question:

> **Which CNN design choices actually help on CIFAR-10, and which ones just sound good on paper?**

Ablation studies are useful because they force each decision to stand on its own. That makes it easier to see the tradeoffs behind learning stability, generalization, training cost, and architectural efficiency.

---

## 🏆 Key Results

Some of the strongest outcomes in the reported runs came from:

- ✅ Increasing training to **10 epochs**
- ✅ Using **SGD with momentum**
- ✅ Reducing dropout to **0.3**
- ✅ Replacing max pooling with **AveragePooling2D**
- ✅ Removing one pooling layer
- ✅ Using larger **(5×5)** convolution filters

### Best reported validation results

| Experiment | Validation Accuracy |
|-----------|---------------------:|
| **10 epochs** | **0.5914** |
| **SGD + momentum** | **0.5447** |
| **Dropout 0.3** | **0.5327** |
| **Remove one pooling layer** | **0.5045** |
| **AveragePooling2D** | **0.5022** |
| **Larger filters (5×5)** | **0.4901** |

One of the clearest takeaways from the study: **training longer helped more than many of the architectural tweaks.**

---

## 📌 Main Findings

### 1) Data augmentation helped — but not all augmentation helped
- `RandomContrast` gave the strongest result among the newly tested augmentation variants.
- `RandomBrightness` performed terribly and stayed near chance level.
- Removing augmentation reduced validation performance, which supports the idea that augmentation still improved generalization overall.

### 2) Preserving spatial information mattered
- `AveragePooling2D` outperformed `MaxPooling2D`.
- Removing one pooling layer improved validation accuracy even further.

This suggests the original architecture may have compressed spatial detail too aggressively for **32×32 CIFAR-10 images**.

### 3) Bigger was not automatically better
- Increasing filter counts to `(64, 128, 256)` made training slower and validation worse.
- Switching from `(3×3)` to `(5×5)` filters improved validation accuracy, though with added training cost.

So no, “more capacity” did not just magically fix things.

### 4) Moderate regularization beat heavy regularization
- `Dropout = 0.3` improved both training and validation accuracy.
- `Dropout = 0.7` clearly underfit.
- `L2 regularization` reduced validation accuracy in the reported run.

### 5) Optimization choices had a huge effect
- `SGD + momentum` gave the best validation accuracy among the tested optimizers.
- `RMSprop` achieved stronger training accuracy but generalized worse.
- Poor learning-rate choices caused near-chance behavior.

### 6) More epochs helped the most
- Moving from **2 epochs to 10 epochs** produced the biggest reported improvement.
- Training curves suggested only mild overfitting later in training.

---

## 📋 Experiment Summary

| Experiment | Training Accuracy | Validation Accuracy | Observation |
|-----------|------------------:|--------------------:|------------|
| RandomContrast | 0.4345 | 0.4408 | Best added augmentation variant |
| RandomBrightness | 0.1005 | 0.1000 | Too disruptive, failed to learn |
| RandomTranslation | 0.3940 | 0.4341 | Helped with positional robustness |
| No augmentation | 0.3953 | 0.4049 | Generalization dropped |
| More filters (64,128,256) | 0.3964 | 0.3183 | Slower and worse |
| Larger filters (5×5) | 0.4153 | 0.4901 | Better features, higher cost |
| AveragePooling2D | 0.4408 | 0.5022 | Better than max pooling |
| Remove one pooling layer | 0.4217 | 0.5045 | More spatial detail helped |
| Dropout 0.3 | 0.4820 | 0.5327 | Best regularization setting tested |
| Dropout 0.7 | 0.3240 | 0.3252 | Clear underfitting |
| L2 regularization | 0.4426 | 0.4266 | Did not help here |
| SGD | 0.4926 | 0.5041 | Strong baseline optimizer |
| SGD + momentum | 0.4973 | 0.5447 | Best optimizer tested |
| RMSprop | 0.5247 | 0.4731 | Strong training fit, weaker generalization |
| 10 epochs | 0.5669 | 0.5914 | Biggest overall improvement |
| Batch size 32 | 0.4219 | 0.4761 | Better generalization, slower |
| Batch size 128 | 0.4226 | 0.4440 | Faster, but weaker validation |

> All values above come from the accompanying written experiment report.

---

## 🔍 Error Analysis

The error analysis showed that many mistakes came from:

- low-resolution ambiguity
- visual similarity between classes
- weak separation between some animal and vehicle categories

### Common confusion patterns
- **truck → automobile**
- **airplane → ship**
- several animal-class mix-ups

A particularly interesting class-level result was for **cat**:

- **Precision:** `0.4751`
- **Recall:** `0.1810`

That usually means the model was relatively cautious when predicting cats, but still missed a lot of true cat images.

The confusion matrix and training-history plots tell a pretty consistent story:

- the model improved steadily during training
- visually similar classes caused the heaviest mistakes
- mild overfitting appeared later in training

---

## 📈 Why This Repo Matters

This project is useful because it shows something more honest than a polished final metric.

It shows:

- what helped
- what hurt
- what looked promising but failed
- what tradeoffs showed up between speed, fit, and generalization

That makes it a stronger learning artifact than just posting a notebook with one final model and calling it done.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

---

## 📂 Repository Structure

You can organize the repo like this:

```bash
cifar10-cnn-ablation-study/
│
├── notebooks/
│   └── cifar10_ablation_study.ipynb
│
├── reports/
│   └── experiment_report.pdf
│
├── images/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── README.md
└── requirements.txt
