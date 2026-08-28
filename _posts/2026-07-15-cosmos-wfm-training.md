---
title: "Cosmos WFM (2/3): Training — Diffusion, Autoregressive, and Post-Training Recipes"
date: 2026-07-15 14:30:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [cosmos, world-foundation-model, diffusion, flow-matching, autoregressive, grpo, model-merging, nvidia, dit]
math: true
image:
  path: /assets/img/ran/cosmos-diffusion-ar-hero.png
  alt: "A robot arm lifting a brain from a sandbox — learning world dynamics in a simulated environment"
---

> **TL;DR**: Cosmos builds world models in two paradigms. Diffusion models generate video by iteratively denoising continuous latent tokens — flow matching (velocity prediction) replaces the earlier EDM (score matching) formulation. Autoregressive models treat video generation as next-token prediction over discrete tokens, exactly like GPT for language. Post-training follows the LLM playbook: supervised fine-tuning per domain → model merging (TIES/DARE) to recombine specialists → RL via GRPO with a VLM-based reward model → timestep distillation for faster inference.

> These paper reviews are written more for me and less for others. LLMs have been used in formatting
{: .prompt-tip }

> This is Part 2 of a 3-part series on NVIDIA's Cosmos World Foundation Models. See [Part 1]({% post_url 2026-07-01-cosmos-wfm-foundations %}) for what WFMs are, data curation, and tokenization. This series draws from [Cosmos WFM Platform (2025)](https://arxiv.org/abs/2501.03575) and [World Simulation with Video Foundation Models (2026)](https://arxiv.org/abs/2511.00062).
{: .prompt-info }

---

## Two Paradigms, One Goal

Both diffusion and autoregressive models solve the same problem: generate high-quality, physics-consistent video. They differ in how they decompose the generation task:

- **Diffusion**: start from pure noise, iteratively refine toward a clean video. The difficulty is distributed across denoising steps.
- **Autoregressive**: generate one video token at a time, left to right. The difficulty is distributed across sequential predictions.

Both use transformer backbones. Both scale to 14B+ parameters. Both support three generation modes — Text2World (text → video), Image2World (text + reference image → video), and Video2World (text + past video → future video).

---

## Diffusion: From Noise to Video

### Score Matching (EDM)

The earlier formulation uses the Elucidated Diffusion Model (EDM). The model is a denoiser $D\_\theta$ that takes a corrupted sample $\mathbf{x}\_0 + \mathbf{n}$ and predicts the clean sample $\mathbf{x}\_0$:

$$\mathcal{L}(D_\theta, \sigma) = \mathbb{E}_{\mathbf{x}_0, \mathbf{n}} \left[\|D_\theta(\mathbf{x}_0 + \mathbf{n};\ \sigma) - \mathbf{x}_0\|_2^2\right]$$

where $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$. The overall loss integrates across noise levels with adaptive weighting — treating the multi-noise-level problem as multi-task learning.

### Flow Matching

The updated formulation switches to **flow matching** (FM). The model predicts the **velocity** of the diffusion trajectory rather than denoising the corrupted input.

Given data $\mathbf{x}$, noise $\epsilon \sim \mathcal{N}(0, I)$, and timestep $t \in [0, 1]$ from a logit-normal distribution, the interpolated latent is:

$$\mathbf{x}_t = (1 - t)\mathbf{x} + t\epsilon$$

The ground-truth velocity (the direction from data to noise):

$$\mathbf{v}_t = \epsilon - \mathbf{x}$$

The model $\mathbf{u}(\cdot; \theta)$ is trained to predict this velocity:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \epsilon, t} \|\mathbf{u}(\mathbf{x}_t, t, \mathbf{c};\ \theta) - \mathbf{v}_t\|^2$$

**Why switch?** Flow matching provides a more direct training target (velocity vs denoised output) and tends to yield smoother optimisation. The interpolation $\mathbf{x}\_t = (1-t)\mathbf{x} + t\epsilon$ defines a straight path between data and noise — conceptually cleaner than EDM's curved noising schedule.

**Timestep scheduling.** Sampling $t$ uniformly would oversample easy noise levels. A shifted logit-normal distribution biases training toward harder, noisier examples — with the shift increasing at higher resolutions.

---

## The DiT Architecture

Both paradigms use a **Diffusion Transformer (DiT)** — repeated blocks of:

1. **Self-attention** over spatiotemporal token positions
2. **Cross-attention** between latent tokens and text embeddings (for text conditioning)
3. **Feed-forward MLP**

Each normalisation layer is modulated by **Adaptive Layer Normalisation (AdaLN)** — scale, shift, and gate parameters conditioned on the diffusion timestep $t$. This lets the network adjust its behaviour based on the current noise level.

### 3D Positional Encoding

Video tokens have three axes — time, height, width. **3D Rotary Position Embedding (RoPE)** — factorised across T/H/W — encodes relative positions. This enables generalisation to unseen resolutions and sequence lengths, similar to how RoPE helps LLMs handle longer contexts than they were trained on.

### Three Generation Modes

For Image2World and Video2World, **frame replacement** is used: conditioning frames are concatenated with noisy generated frames along the temporal axis, with a binary mask distinguishing them. The denoising loss applies only to the generated frames.

---

## Autoregressive: GPT for Video

The autoregressive WFM treats video generation as next-token prediction. A video is converted into discrete tokens $\mathcal{V} = \{v\_1, v\_2, \ldots, v\_n\}$ using FSQ (vocabulary 64,000). The training objective:

$$\mathcal{L}_{NLL} = \sum_i -\log P(v_i \mid v_1, \ldots, v_{i-1};\ \Theta)$$

At inference, tokens are generated one at a time, each conditioned on all previous tokens — exactly like GPT generating text.

The architecture is a **Llama3-style transformer**: self-attention with 3D RoPE, cross-attention for text conditioning, and SwiGLU feed-forward layers. The deliberate resemblance to LLM architectures means existing infrastructure and optimisation techniques transfer directly.

### The Compression-Quality Gap

Discrete tokenization at aggressive compression ($8 \times 16 \times 16$) loses detail — outputs can be blurry. To fix this, a **diffusion decoder** enhances the discrete tokens back to high quality. The discrete token video is upsampled and concatenated with noisy continuous tokens, then a diffusion model is fine-tuned to denoise conditioned on the discrete tokens. This is a learned super-resolution step that recovers detail lost during compression.

---

## Post-Training: The LLM Playbook Applied to Video

This is where the most transferable ideas live. The post-training recipe mirrors what worked for language models.

### Step 1: Supervised Fine-Tuning (SFT)

After pre-training, fine-tune separate models on domain-specific data — one for object permanence, one for high-motion scenes, one for driving, one for robotics, etc. Each specialist improves on its target domain.

### Step 2: Model Merging

Fine-tuning separate models per domain raises a question: how do you get a single model that is good at everything? Rather than multi-task training (expensive, tricky to balance), you merge the weights of multiple fine-tuned models.

Three methods:
- **Model Soup**: simple weight averaging — $\theta\_{\text{merged}} = \frac{1}{N}\sum\_i \theta\_i$
- **TIES**: trim small-magnitude weight changes, resolve sign conflicts across models, then merge. This removes noise from the merge.
- **DARE**: randomly drop a fraction of delta weights (the changes from pre-trained to fine-tuned), then average. This prevents interference between specialists.

All three achieve comparable performance. Model Soup is simplest. The merged model gets the best of all worlds — improved domain-specific performance while maintaining general quality.

This is a powerful idea: train cheap specialists, merge them for free.

### Step 3: RL with GRPO

The final alignment step applies RL to match generation quality with human preferences — the video equivalent of RLHF for language.

**Reward model**: a VLM-based scorer that rates generated videos on text alignment (does the video match the prompt?), motion quality (is motion natural?), and visual quality (sharp, well-lit, artefact-free?).

**Algorithm**: GRPO (Group Relative Policy Optimisation). For each input:
1. Generate 8 candidate videos
2. Score each with the reward model
3. Normalise rewards within the group to compute advantages
4. Update the model to increase probability of high-advantage outputs

The diffusion model's denoising trajectory is treated as a sequence of states and actions — the RL gradient flows through the denoising steps.

**Regularisation**: the standard diffusion loss is added as a regulariser to prevent reward hacking (the model gaming the reward model at the expense of actual quality).

**Result**: RL-trained models are preferred by humans ~46% vs ~16% over pre-RL models, with ~37% ties.

### Step 4: Timestep Distillation

Diffusion models typically need 20-50 denoising steps. Distillation compresses this to **4 steps** with comparable quality, using a hybrid framework that combines consistency distillation with distribution matching. This is critical for practical deployment — 5x fewer steps means 5x faster inference.

---

## What Is Coming in This Series

- **[Part 1 — Foundations]({% post_url 2026-07-01-cosmos-wfm-foundations %})**: What WFMs are, data curation, and video tokenization.
- **[Part 3 — Applications]({% post_url 2026-07-29-cosmos-wfm-applications %})**: Sim2Real, robot policy augmentation, action-conditioned generation, and multi-view simulation.

---

## Key Takeaways

- **Diffusion models** denoise continuous latent tokens; **autoregressive models** predict discrete tokens sequentially. Both are valid approaches to world simulation.
- Flow matching (velocity prediction) replaces score matching (EDM) — straight interpolation paths, smoother optimisation.
- The DiT architecture is shared: self-attention + cross-attention + FFN, modulated by AdaLN conditioned on noise level.
- 3D RoPE encodes spatiotemporal positions, generalising across resolutions and lengths.
- Autoregressive models are deliberately Llama3-style for infrastructure reuse. A diffusion decoder recovers quality lost to aggressive discrete compression.
- Post-training follows the LLM playbook: SFT per domain → model merging (cheap way to combine specialists) → RL with GRPO (align with human preferences) → distillation (20 steps → 4).
- Model merging is the sleeper hit: train separate specialists, average their weights, get a model better than any individual — for free.
