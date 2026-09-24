---
title: "Data Scaling Laws in Imitation Learning"
date: 2026-09-20 14:30:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [robotics, imitation-learning, scaling-laws, diffusion-policy, data-collection, generalization]
math: true
---

> **TL;DR**: How should you spend a data-collection budget for a robot policy? This paper's answer: on *diversity*, not volume. Zero-shot performance in new environments and on new objects follows a rough power law in the number of training environments and objects. It does not follow one in the number of demos, which plateaus quickly. Their recipe: 32 environment-object pairs × 50 demos each gets a single-task policy to ~90% success in unseen places with unseen objects. That's one afternoon of work for four people.

> These paper reviews are written more for me and less for others. LLMs have been used in formatting
{: .prompt-tip }

> This post draws from [Data Scaling Laws in Imitation Learning for Robotic Manipulation (Lin et al., ICLR 2025)](https://arxiv.org/abs/2410.18647).
{: .prompt-info }

---

## The Question

Can the right data give you a single-task policy that works on *any* object in a category, in *any* environment, with no fine-tuning?

They split generalisation into two axes:
- **Environment**: new lighting, backgrounds, distractors, table surfaces
- **Object**: new instances of the same category (a different bottle, a different mouse)

Generalising to new *tasks* is explicitly out of scope, because it would need data from thousands of tasks.

Unlike prior work, they don't isolate single factors (only the lighting colour, or 3D-printed objects that vary only in size). They use real in-the-wild environments and everyday objects, where everything varies at once. That is closer to what deployment actually looks like.

---

## Setup

- **Data**: human demos collected with [UMI](https://umi-gripper.github.io/), a handheld gripper with a GoPro. No robot is needed during collection, so moving between environments is cheap.
- **Policy**: Diffusion Policy with a fully fine-tuned DINOv2 ViT-L/14 encoder.
- **Tasks**: Pour Water and Mouse Arrangement for the main study; Fold Towels and Unplug Charger for validation.
- **Scale**: 40,000+ demos and 15,000+ real-robot rollouts.

The evaluation protocol deserves a mention. They score each stage of a task (0–3 points per step) instead of using binary success, because binary success is too sparse to tell policies apart. They also run blind tests: all policies are evaluated from the same initial conditions, in a shuffled order.

---

## The Findings

Three experiments: vary the objects (in one environment), vary the environments (with one object), then vary both together. Each experiment also varies the fraction of demos used.

**1. Performance follows a power law in environments and objects.** On log-log axes, the optimality gap (1 − score) against the number of environments or objects is close to a straight line, with correlation 0.94–0.99.

**2. Performance does not follow a power law in the number of demos.** Adding demos helps at first, then flattens out. With 16 environments and 64 objects, performance plateaus around 800 total demos. The correlation for demos is only −0.62 to −0.79.

**3. More diversity means fewer demos per environment.** With 8 training objects, using 12.5% of the demos is clearly worse than using 100%. With 32 objects, that gap almost disappears. When environment *and* object both vary, the demo count saturates even faster.

**4. Object generalisation is easier than environment generalisation.** With only 8 training objects, the score on unseen objects is already above 0.8. The environment curve rises more slowly.

The advantage from diversity also holds when the total number of demos is fixed (Appendix G.2). So it isn't just a side effect of having more data.

They extrapolate that reaching 0.99 on Mouse Arrangement would take ~1,191 environment-object pairs. They fit only 6 points and say so, and they deliberately don't fit an irreducible-error term.

---

## The Recipe

- **Collect in as many environments as possible, with one object per environment.** Putting several objects in each environment helps when you have few environments. By ~16 environments the benefit disappears.
- **Collect about 50 demos per environment-object pair.** For 8, 16 and 32 pairs, performance plateaus at 400, 800 and 1,600 demos respectively.
- **32 pairs is enough** for tasks of this difficulty.

To validate the recipe, they applied it to two new tasks. Four collectors worked for one afternoon, and all four tasks reached 85–92% success across 8 unseen environments with unseen objects.

---

## The Model Side

These are small ablations on Pour Water, and they're worth knowing:

| Visual encoder | Score |
|---|---|
| DINOv2 ViT-L/14, full fine-tune | 0.90 |
| DINOv2, LoRA (rank 8) | 0.72 |
| ViT-L/14 from scratch | 0.03 |
| DINOv2, frozen | **0.00** |

- **You need both pretraining and full fine-tuning.** A frozen encoder completely fails.
- **Scaling up the encoder helps steadily**: ViT-S 0.66 → ViT-B 0.81 → ViT-L 0.90.
- **Scaling up the action diffusion U-Net doesn't help**, and the largest one does slightly worse. Either a small head is already enough, or nobody has found an action-head architecture that scales yet.

---

## Appendix Nuggets

**Validation MSE is not a reliable metric.** In some settings it correlates almost perfectly with real performance (r = −0.98). In others it doesn't: MSE *rose* at 16 objects while real performance kept improving. LoRA had *lower* MSE than full fine-tuning (0.0049 vs 0.006) but scored worse on the robot. The authors use MSE only to catch obviously broken policies.

**A cheap fix for ambiguous states.** With the standard 2-frame observation window (0.05s), the Pour Water policy can't tell whether it is *about to* pour or has *just finished* pouring, because the bottle is at the mug's mouth either way. Adding one frame from 0.25s earlier fixes this. It costs almost nothing and the idea carries over to other tasks.

**Nested training sets.** Larger training sets always contain the smaller ones, which keeps the data distribution consistent as the size changes. Training epochs are also scaled so that small-data policies still converge.

**Data-collection lessons from UMI:**
- Randomise the gripper's starting height and orientation. Otherwise the policy only works from the poses it was trained on.
- Get collectors to behave the same way and finish in similar times, to cut down on multimodal behaviour in the data.
- Large objects that fill the camera view (doors, drawers) break SLAM tracking. That's why these tasks were avoided.
- Roughly 90% of demos survive SLAM filtering.

---

## Limitations

- Single-task policies only. Generalising to new tasks is the next question.
- Imitation learning only, with Diffusion Policy only.
- Four tasks, all of moderate difficulty. Dexterous tasks probably need more than 50 demos per pair.

---

## Key Takeaways

- Zero-shot performance follows a power law in *environments and objects*, not in *demos*.
- Diversity beats volume. Once you have enough environments, extra demos per environment add little.
- Practical recipe: many environments, one object each, ~50 demos each. 32 pairs was enough here.
- The vision encoder matters more than the action head: pretrain it, fully fine-tune it, and make it big.
- Don't trust validation MSE as a proxy for how the robot will perform.
