---
title: "ROBOMETER: Scaling General-Purpose Robotic Reward Models"
date: 2026-08-12 14:30:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [robometer, reward-model, robotics, vlm, preference-learning, physical-ai, reinforcement-learning]
math: true
image:
  path: /assets/img/ran/robometer-hero.png
  alt: "ROBOMETER architecture — robot trajectories flow into a VLM backbone producing task progress, success, and trajectory comparisons"
---

> **TL;DR**: Zero-shot reward tricks (like TOPReward) are elegant but limited. ROBOMETER asks: what happens if you actually *train* a reward model at scale? Built on Qwen3-VL-4B with three lightweight MLP heads (progress, success, preference), trained on 1M trajectories across 21 robot embodiments, it produces dense per-frame rewards that outperform all baselines — enabling online RL that reaches 85% success where the next best method plateaus at 55%, and data retrieval that yields 4.5x higher downstream policy success.

> These paper reviews are written more for me and less for others. LLMs have been used in formatting
{: .prompt-tip }

> This post draws from [ROBOMETER — Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons](https://arxiv.org/abs/2603.02115).
{: .prompt-info }

---

## From Zero-Shot to Trained

Zero-shot reward methods like [TOPReward]({% post_url 2026-08-05-topreward %}) extract reward signals from off-the-shelf VLMs without any training. They work surprisingly well for what they are — but they hit a ceiling. A log-probability of "True" can track coarse progress, but it can't reliably distinguish between a robot that *almost* completed a task and one that completed it badly, or between a trajectory that matches the instruction and one that manipulates the wrong object.

ROBOMETER's bet: if you train a VLM-based reward model on diverse robotics data with the right objectives, you can break through that ceiling while staying general-purpose — one model, many embodiments, many tasks, no per-task engineering.

---

## Architecture: Three Heads on a Frozen VLM

The backbone is **Qwen3-VL-4B-Instruct** — a causally masked vision-language model that processes interleaved text and image tokens. On top of its hidden states, three lightweight 2-layer MLP heads are attached:

**Progress head.** Applied per-frame. Outputs a categorical distribution over 10 bins covering $[0, 1]$. At inference, the continuous progress estimate is the expected value across bin centers. Why discretize instead of regressing a scalar? Following the distributional RL literature — binned classification better captures multi-modal reward distributions than a single regression target.

**Success head.** Applied per-frame. Binary classifier — is the task complete at this point? Gives a crisp done/not-done signal.

**Preference head.** Applied once per trajectory pair. A single logit indicating which of two trajectories better accomplishes the task. This is the comparative judgment — "A is better than B" — which is easier for both humans and models than absolute scoring.

---

## The Tokenization Trick

The clever part is how two trajectories are packed into a single forward pass. The prompt contains both trajectories sequentially, with special tokens controlling what each head can attend to:

- **Progress tokens** are inserted after every frame of Trajectory A only. Thanks to the causal mask, each progress token at time $t$ attends only to frames $1$ through $t$ — giving dense, online-usable, frame-level progress without peeking at the future.
- **A split token** separates the two trajectories.
- **A preference token** is appended after both trajectories — it attends to everything, enabling the cross-trajectory comparison.

Why not put progress tokens in Trajectory B too? They'd end up attending to Trajectory A as well (breaking the "progress = single trajectory" semantics), and at inference you only ever need progress for one trajectory anyway.

This design means the preference head gets to do actual *cross-video comparison* within a single forward pass — tokens from one trajectory directly attend to the other. This outperforms the Bradley-Terry alternative (score each trajectory independently, couple only through the loss) by a wide margin.

---

## Training Data: RBM-1M

A dataset of ~1.06 million trajectories across 21 robot embodiments — from Franka Pandas to humanoids to simulated arms. The data comes in three flavors:

**Expert demonstrations** — successful task completions with progress target 1.0. These teach the model what "done" looks like.

**Mixed expertise** — paired success/failure trajectories from the same task. These are the preference-learning fuel — the model sees two attempts at the same goal and learns which is better.

**Human-only data** — e.g., Epic-Kitchens clips used only for preference training (noisy labels, but still useful signal).

---

## Three Data Sampling Strategies

At training time, each example is built by first sampling a *strategy*:

**Progress-based comparisons.** Pair a successful trajectory with a failed one on the same task. The preference target is obvious (success wins), and critically, *the failure trajectory needs no progress labels* — just being worse than the expert is enough signal.

**Instruction negatives.** Pair two trajectories with *different* instructions. Pick one instruction as the conditioning — the trajectory matching it wins; the other trajectory's progress is forced to zero (correct behavior for the wrong task = no reward). This teaches the model to ground rewards in the actual instruction, not just "does the robot move smoothly."

**Video rewind.** From a single expert trajectory, reverse a segment to create a synthetic failure. The forward version wins the preference; the rewound version gets decreasing progress targets. This is free data augmentation — no additional collection needed.

---

## Why Preference Helps

The ablation story is clean:

| Training Signal | Kendall $\tau$ (LIBERO-90) |
|----------------|---------------------------|
| Progress only | 0.63 |
| + Preference | 0.74 |
| + Preference + failure data | **0.92** |

Progress-only training (just predicting how far along a trajectory is) gives decent per-trajectory tracking but weak *ranking* between trajectories. Adding the preference objective — even without any failure data — improves ranking substantially. Adding actual failure data on top produces the largest single gain.

A further ablation confirms the VLM backbone matters: replacing Qwen3-VL with a small transformer trained from scratch (same objectives, same data) collapses to near-zero Kendall $\tau$. The pre-trained visual understanding is load-bearing.

---

## Downstream Applications

This is where ROBOMETER earns its keep — four distinct use cases:

### Online RL

Pair ROBOMETER with a policy (e.g., $\pi_0$) and run RL in the real world. On a two-stage manipulation task: ROBOMETER drives success from 20% → 70%, while the next best reward model (RoboReward) stalls at 20% → 20%. RoboReward's failure mode: it assigns near-max reward to wrong-object manipulation (poor instruction grounding), causing premature resets that reinforce bad behavior.

### Offline RL

Use ROBOMETER to relabel a mixed-quality dataset with dense rewards, then run IQL. ROBOMETER performs best at *lower* discount factor ($\gamma = 0.9$) — its dense, temporally-aligned rewards reduce reliance on long-horizon credit assignment. Result: 2.4x the success rate of the best baseline, averaged across tasks.

### Data Filtering and Retrieval

Given a large unstructured "play" dataset (robots doing various things), use ROBOMETER to retrieve the most task-relevant, high-quality subtrajectories for a target task. Downstream policies fine-tuned on ROBOMETER-retrieved data achieve **4.5x higher success rate** than the best baseline. The baselines retrieve topically-relevant but low-quality (failed/suboptimal) trajectories — ROBOMETER's dual progress + preference signal filters for both relevance *and* quality.

### Failure Detection

ROBOMETER detects failures zero-shot by looking for temporal inconsistency in its own predicted reward curve — a trajectory where progress stalls, oscillates, or drops signals failure. It catches both catastrophic failures (drops, spills) and subtle ones (stalling, oscillation). Highest average F1 across tested methods, no per-task calibration needed.

---

## Head-to-Head: ROBOMETER vs TOPReward

TOPReward is a direct baseline in this paper. The gap is significant:

| Method | VOC (ID) | VOC (OOD) | Kendall $\tau$ (OOD) |
|--------|----------|-----------|---------------------|
| TOPReward | 0.42 | 0.40 | 0.13 |
| **ROBOMETER** | **0.92** | **0.94** | **0.64** |

A purpose-trained model with progress + preference supervision on 1M curated trajectories substantially outperforms a pure zero-shot logit trick. The VOC numbers aren't directly comparable across the two papers (Pearson here vs Spearman in TOPReward's paper), but the gap is real.

That said, TOPReward's value is its zero-shot nature — no data collection, no training. ROBOMETER needs a million trajectories. They sit at different points on the effort-vs-quality tradeoff.

---

## Fine-Tuning ROBOMETER on New Domains

When deployed on a new domain-specific dataset (RoboFAC: 16 tasks, 11K trajectories), fine-tuning from ROBOMETER's checkpoint — even with just LoRA — dramatically outperforms fine-tuning the base Qwen3-VL from scratch:

| Starting Point | Kendall $\tau$ |
|---------------|---------------|
| Qwen3-VL from scratch (full FT) | 0.10 |
| **ROBOMETER checkpoint (LoRA)** | **0.79** |

The base VLM learns to track per-trajectory progress but fails to *rank* trajectories against each other. ROBOMETER's pre-training instills the comparative judgment that transfers even with lightweight adaptation.

---

## Key Takeaways

- A trained VLM-based reward model (progress + success + preference heads on Qwen3-VL-4B) substantially outperforms zero-shot methods across the board.
- The tokenization design — progress tokens in one trajectory only, a preference token attending to both — enables cross-video comparison in a single forward pass, beating the Bradley-Terry alternative.
- Three data sampling strategies (progress comparisons, instruction negatives, video rewind) make failure data useful without requiring progress labels for failures.
- Preference training is the key ingredient — it improves trajectory *ranking* even without failure data, and the gain compounds when failure data is added.
- Downstream, ROBOMETER enables online RL (2.5x better than next best), offline RL (2.4x), data retrieval (4.5x), and zero-shot failure detection.
- ROBOMETER's checkpoint transfers well to new domains via lightweight fine-tuning — the comparative judgment it learned is general.
