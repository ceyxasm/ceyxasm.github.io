---
title: "TOPReward: Zero-Shot Robot Rewards from Token Probabilities"
date: 2026-08-05 14:30:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [topreward, reward-model, robotics, vlm, zero-shot, physical-ai]
math: true
image:
  path: /assets/img/ran/topreward-hero.png
  alt: "A robot arm folding a towel with a True_Logit probability readout — the core TOPReward concept"
---

> **TL;DR**: Need a reward signal for robot learning but don't want to train a reward model? Prompt a VLM with "this trajectory completes the task: True or False?" and use the log-probability of the "True" token as the reward. No training, no fine-tuning — just a single forward pass per frame. This zero-shot trick produces progress curves that track task completion surprisingly well, enabling offline RL improvements on real robots.

> These paper reviews are written more for me and less for others. LLMs have been used in formatting
{: .prompt-tip }

> This post draws from [TOPReward — Token Probabilities as Hidden Zero-Shot Rewards for Robotics](https://arxiv.org/abs/2602.19313).
{: .prompt-info }

---

## The Problem: Reward is the Bottleneck

Robot learning — whether RL, offline RL, or weighted imitation — needs a reward signal. Sparse rewards (binary success/fail at the end) work but are sample-inefficient. Dense rewards (continuous progress at every step) are far better for learning but traditionally require either hand-engineering or training a separate reward model on labelled data.

The question TOPReward asks: can you extract a dense, per-step reward from a pre-trained VLM without any training at all?

---

## The Core Trick

Take a VLM (e.g., Qwen3-VL-8B). Show it a video prefix — the first $t$ frames of a robot trajectory. Append the prompt:

> "The above video shows a robot manipulation trajectory that completes the following task: {INSTRUCTION}. Decide whether the above statement is True or not. The answer is: True"

Now read off the log-probability the model assigns to that final "True" token:

$$r_t = \log\, p_\theta(\text{"True"} \mid \text{video}_{1:t},\ \text{prompt})$$

That's the reward. At the start of the trajectory (few frames, task barely started), the VLM has little visual evidence that the task is complete — $p(\text{"True"})$ is low, $r\_t$ is a large negative number. As the robot makes progress, the visual evidence accumulates — $p(\text{"True"})$ rises, $r\_t$ approaches zero. At task completion, the statement is essentially correct and $r\_t \approx 0$.

You get a monotonically increasing progress curve for free.

---

## Why This Works (and What Doesn't)

The key insight is that this sidesteps two known weaknesses of VLMs:

**No generation required.** You're not asking the model to output a well-formatted number ("rate progress from 0 to 1"). You're just reading a probability the model already computes during its normal next-token prediction. This is a much lower bar — the model doesn't need to be good at instruction-following or numeric formatting.

**No frame shuffling.** Prior work (GVL) asks the VLM to look at shuffled frames and rank them — a harder reasoning task that requires the model to overcome its tendency to assign progress based on frame position rather than content. TOPReward shows frames in chronological order and lets the natural accumulation of visual evidence do the work.

**The rejected alternative.** The authors also tried scoring the probability of the *instruction text itself* given the video — "how likely is it that this video is described by these words?" This fails because the VLM assigns high probability to instruction tokens (e.g., "apple") simply because the object is *visible*, regardless of whether the task involving it has been *completed*. The binary True/False framing forces an explicit completion judgment.

**Why "True" specifically?** They compared candidate tokens by measuring which one shows the cleanest separation in mean probability between successful and failed trajectories at the final frame. "True" had the largest gap and is a single token (avoiding multi-token aggregation issues).

---

## From Raw Probabilities to Usable Rewards

The raw log-probability $r\_t$ lives in $(-\infty, 0]$. Two post-processing steps make it practical:

**Per-episode normalization.** Min-max normalize across the sampled timesteps within each episode to get a $[0, 1]$ progress score. This makes the curve interpretable but means absolute values aren't comparable across different episodes — a known limitation.

**Dense per-step reward.** For downstream RL/IL, compute a per-transition reward as a clipped exponential of the progress delta:

$$\Delta_{t_k} = \text{clip}\left(\tau \cdot \exp(s_{t_k} - s_{t_{k-1}}),\ 0,\ \delta_{\max}\right)$$

The floor at zero means regressions (progress going backward) get zero weight, not negative weight. The ceiling $\delta\_{\max}$ prevents a single high-delta transition from dominating the training objective.

---

## The Chat Template Trap

A surprising finding: wrapping the prompt in a chat template (system/user/assistant turns) instead of feeding it as raw text destroys performance.

| Setup | Mean VOC |
|-------|----------|
| Qwen3-VL-8B, raw text | **0.945** |
| Qwen3-VL-8B, + chat template | 0.500 (−47%) |

The explanation: the raw log-probability trick aligns with the model's *pretraining* objective (plain next-token prediction). A chat template shifts the model into its *instruction-tuned* distribution, which is less well-calibrated for this particular use.

This directly explains why Gemini underperforms in their experiments — Gemini's API forces a chat template with no way to bypass it, structurally disadvantaging it for this method.

---

## Results

**Progress tracking.** On ManiRewardBench (130+ manipulation tasks, 4 robot platforms), TOPReward with Qwen3-VL-8B achieves VOC 0.942–0.954 across all platforms — remarkably tight, suggesting robust generalization. It beats the prior zero-shot method (GVL) by a wide margin (GVL scores 0.16–0.54 depending on backbone).

**Success detection.** VOC (rank correlation) has a structural blind spot — it only measures ordering, not absolute level. A trajectory that makes partial progress but never finishes can still score high on VOC. TOPReward avoids this because its raw signal (probability of *completion*) is naturally depressed for incomplete trajectories. On the failure detection split, TOPReward achieves 0.654 ROC-AUC vs GVL's 0.519 (essentially chance) on Qwen3-VL.

**Real-world RL.** Using TOPReward's dense rewards for advantage-weighted regression (AWR) on a real SO-100 robot arm — 50 demonstrations per task, potentially noisy:

| Task | Pretrained | BC | TOPReward-AWR |
|------|-----------|-----|---------------|
| Place toy car in box | 1/10 | 2/10 | 3/10 |
| Place doll in box | 0/10 | 7/10 | **10/10** |
| Put cube in cup | 4/10 | 6/10 | **9/10** |

TOPReward-AWR beats plain BC (same data, unweighted loss) on all six tasks. Notably, BC sometimes *hurts* relative to the pretrained baseline — noisy demonstrations can actively degrade naive imitation, while advantage-weighting via TOPReward filters toward useful transitions.

---

## Limitations

- Inherits the visual perception limits of the underlying VLM — fine-grained spatial reasoning (precise alignment, small objects) gets noisy progress estimates.
- Per-episode normalization prevents direct comparison of absolute progress across different trajectories.
- The method's ceiling is bounded by the VLM's video understanding — but this is also a feature: future VLM improvements translate directly into better rewards with zero method changes.

---

## Key Takeaways

- Log-probability of a single "True" token, conditioned on video + a completion statement, produces a surprisingly effective zero-shot dense reward for robot learning.
- No training, no fine-tuning — just a forward pass through an off-the-shelf VLM.
- The trick works because it avoids asking the VLM to *generate* calibrated numbers and instead reads a probability the model already computes.
- Chat templates destroy the signal — use raw text prompts for this method.
- Practical enough for real-world RL: advantage-weighted regression with TOPReward rewards improves over both pretrained baselines and naive BC on real robot tasks.
