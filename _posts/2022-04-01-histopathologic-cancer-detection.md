---
title: "Why CNNs Work: Histopathologic Cancer Detection"
date: 2022-04-01
categories: [Research, Machine Learning]
tags: [deep-learning, cnn, medical-imaging, dimensionality-reduction, pca, pytorch, cancer-detection]
---

This was a course project at IIT Jodhpur — but one I found genuinely interesting to work through, because it forced a question: before training a neural network, can you actually *see* in the data that the signal is there?

The task: binary classification of histopathology images as cancerous or not, on the [Kaggle histopathologic cancer detection dataset](https://www.kaggle.com/c/histopathologic-cancer-detection) (220k training images).

Done with Anirudh Srikanth and Atharva Pandey.

## Phase 1: Does more data always help?

We worked with a stratified subsample of 5,500 images to build the initial pipeline. The full 220k was used only for the final CNN run. The key insight from sampling: increasing data volume doesn't linearly increase information, especially when the distribution is already captured by a smaller representative sample.

The dataset was mildly imbalanced: 59.5% positive, 40.5% negative — manageable without heavy resampling.

## Phase 2: Confirm the signal exists

Before any model, we ran clustering analysis via dendrograms on dimensionality-reduced data. Two clearly separable clusters emerged — confirming the cancerous/non-cancerous distinction is visible even in compressed representations. This is a useful sanity check that's often skipped.

## Phase 3: How much can you compress?

Each image is 96×96×3 = 27,648 raw features. PCA analysis showed:
- **95% variance** captured with only **2,432 features**
- **98% variance** captured with only **3,400 features**

That's a 9× reduction with almost no information loss. LDA on top of PCA further improves classical model performance by making the compressed space discriminative rather than merely reconstructive.

## Classical ML Results

Best result: **Linear SVC + PCA (0.9 variance threshold) + LDA → F1 = 0.83**

Without dimensionality reduction, LightGBM and RBF SVC peaked around F1 = 0.78. Reducing dimensions first genuinely helped — the curse of dimensionality is real in 9,216-dimensional space.

## CNN Architecture

5 sequential convolutional blocks, each: Conv2D → BatchNorm → ReLU → MaxPool2d

Channels double per block: 32 → 64 → 128 → 256 → 512

Then: AvgPool2d → Linear(512 → 2)

### Results
- Trained on 5,500-image subsample: **92.27% validation accuracy**
- Trained on full 220k for 10 epochs: **94.64% test accuracy**

The CNN outperforms every classical approach — and not by a small margin.

## Why CNNs win here

The thing worth noting: a CNN's architecture *is* dimensionality reduction. Every conv+pool layer is discarding spatial information while preserving the features that matter most for classification. PCA does this globally and linearly; a CNN does it locally and nonlinearly, and learns *what* to discard from the task itself rather than from reconstruction error.

Full report [here](https://github.com/ceyxasm/histopathologic-cancer-detection/blob/main/report.pdf) — GitHub repo [here](https://github.com/ceyxasm/histopathologic-cancer-detection).
