---
title: "Sim-and-Real Co-Training: A Simple Recipe"
date: 2026-09-24 09:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [robotics, sim-to-real, co-training, imitation-learning, mimicgen, diffusion-policy]
math: true
---

> **TL;DR**: Skip sim-to-real transfer. Train one policy on a *mix* of a small real dataset and a large simulated one. Across 6 tasks on a Panda arm and a GR-1 humanoid, adding sim data raises average real-world success from 45% to 83%. This works even when the sim data is poorly aligned with the real task, as long as there's a lot of it and you tune the mixing ratio. The best ratio was 90–99% sim. What has to match is task semantics, the camera viewpoint (roughly), and the *behaviour patterns*. Physics and visual realism barely matter.

> These paper reviews are written more for me and less for others. LLMs have been used in formatting
{: .prompt-tip }

> This post draws from [Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation (Maddukuri et al., 2025)](https://arxiv.org/abs/2503.24361).
{: .prompt-info }

![Co-training workflow — real demos, digital cousin sim data and prior sim data mixed by ratio α into one policy](/assets/img/ran/sim-real-cotraining-hero.png)
_The workflow: a few real demos, 100× digital-cousin sim data, 1000× prior sim data, mixed per batch by α._

---

## The Idea

Sim-to-real transfer (train in sim, deploy on the real robot) usually needs heavy tuning: system identification, carefully chosen randomisation ranges, digital twins. Co-training avoids most of that. The policy always sees *some* real data, so the sim data only has to be *useful*. It doesn't have to be *faithful*.

The objective is a weighted behaviour-cloning loss:

$$\mathcal{L} = \alpha \cdot \mathcal{L}(\theta; D_{sim}) + (1 - \alpha) \cdot \mathcal{L}(\theta; D_{real})$$

In practice, $\alpha$ is the **probability that each sample in a batch comes from sim**. Since $D\_{sim}$ is orders of magnitude larger than $D\_{real}$, $\alpha$ ends up being the most important knob in the method.

---

## Two Kinds of Sim Data

**Task-agnostic prior data.** Existing sim datasets built without the real task in mind. For the Panda this was RoboCasa (60k demos, 20 kitchen tasks). The only change they made was re-rendering from roughly the real camera pose. Everything else stayed mismatched: robot start pose, controller, objects, physics.

**Task-aware digital cousins.** A sim version of the real task, built with light effort. This borrows the term from [Digital Cousins]({% post_url 2026-09-22-digital-cousins %}), but here it's a hand-built sim *task*, not a scene reconstructed from a photo. It keeps four things the same:
1. The same robot and action space
2. The same task goal and success check (and language instruction)
3. The same object *categories* (instances can differ)
4. The same fixture categories (counters, cabinets, tables)

Scale comes from **MimicGen / DexMimicGen**. Take a few dozen source demos in sim, split them into object-centric segments, then transform and stitch those segments into thousands of new trajectories. Ten human demos become 1,000; 100 become 10,000.

---

## Results

Real demos per task: 50 on the Panda, 20 on the humanoid.

| Training data | Average real success |
|---|---|
| Real only | 45.3% |
| Real + Prior | 76.8% |
| Real + Digital Cousin | 81.1% |
| Real + DC + Prior | **83.2%** |

- **Even the task-agnostic prior data gives +31.5%.** That data was never designed for these tasks.
- Better-aligned cousin data adds more, and combining both is best.
- **CloseDoor**: 10% with real data only vs 100% with any co-training. Doubling the real demos to 100 only brings real-only up to 80%.

**Generalisation beyond the real data.** Sim covers cases the real demos don't:
- **Unseen objects**: Panda 33% → 50%, humanoid 10% → 80%. The humanoid's real data had just one red cup, so the diverse sim objects mattered a lot more there.
- **Unseen positions**: real demos had objects only at the edges of the workspace, and testing put them in the centre. Co-training roughly doubled success.

**It still helps with more real data.** On a 4-task humanoid setup, with real demos varied from 40 to 400, the co-trained policy stays ahead at every point.

---

## What Actually Matters

**You need a lot of sim data.** Cutting cousin demos from 10k to 500 dropped Panda success from 67% to 53%. Cutting from 1k to 100 dropped the humanoid from 95% to 75%.

**The mixing ratio has to be tuned, and the best value is extreme.** A 1:1 split is clearly worse. The best was **α = 0.9–0.99**, which peaks at 95%. Pushing further to 99.5% and 99.9% drops success to 80% and then 60%. At both ends (10% and 99.9% sim), co-training is *worse* than using real data only. The real data is a tiny fraction of each batch, but it can't go to zero.

![Success rate vs co-training ratio α on CupPnP](/assets/img/ran/sim-real-cotraining/cotraining-ratio.png){: w="650" }
_Humanoid CupPnP, 20 real + 1,000 digital-cousin demos. The dashed line is real-only (65%)._

**The camera viewpoint needs to be roughly right.** Rendering cousins from the default sim camera instead of the aligned one cost 11 points on the Panda and 25 on the humanoid. The aligned camera still isn't exact: the real humanoid camera has fisheye distortion and the sim one doesn't. Close enough is enough.

![Real, default-camera and aligned-camera views with co-trained success rates](/assets/img/ran/sim-real-cotraining/camera-alignment.jpg){: w="650" }
_Top: GR-1 CupPnP, 70% → 95% with the aligned camera. Bottom: Panda CounterToSinkPnP, 56% → 67%._

---

## Appendix Nuggets

**Aligning physics didn't matter.** They tuned physics parameters so open-loop sim rollouts matched the real ones. On humanoid CupPnP, success was 95% with or without it.

**How far off the "misaligned" camera was.** The default camera was 37cm and 20° away from the aligned one on the Panda, and 36cm and 60° on the humanoid. That's a large gap, and co-training still helped at the default pose, just less.

**Behaviour patterns must match.** This is the most important caveat. On a bimanual task (pick up a cube with the left hand, hand it over, place it with the right), co-training with *single-arm* prior data at α = 0.99 made the policy do single-arm pick-and-place, and success was near zero. Visual mismatches are tolerated; *behavioural* mismatches are not. With bimanual cousin data, the same task went from 15% to 50%, which also beat real-only with twice the real demos (30%).

**Making sim look realistic helps only at the margins.** They fine-tuned a video diffusion model (CogVideoX) on the real demo videos, then used it to restyle sim videos: add noise to a sim video and denoise it. At noise strength 0.6 the textures look realistic while objects stay where they were (higher strength moves objects and breaks the action labels). The gains were biggest when both real and sim data were scarce. With plenty of sim data, visual realism barely mattered.

**Randomisation and domain adaptation are optional extras.** Adding generated texture randomisation to the cousins helped, but co-training works without it.

---

## The Recipe

- **Sim data**: task-aware cousins are best. Large existing multi-task sim datasets also help, as long as their behaviour matches the real task.
- **Alignment**: the same task definition and success check, and a similar camera viewpoint. Physics and textures can be off.
- **Diversity**: vary objects and positions in sim to cover what the real demos miss.
- **Scale**: orders of magnitude more sim than real data.
- **Ratio**: tune α; expect the best value around 0.9–0.99.

---

## Limitations

- Mostly pick-and-place tasks. High-precision insertion and long-horizon tasks aren't tested.
- Deformable objects and liquids are hard to simulate, so this recipe doesn't apply to them yet. Video and world models might fill that gap.
- Success improves a lot but is still well short of 100%.

---

## Key Takeaways

- Co-training sidesteps sim-to-real transfer: the policy always sees some real data, so sim only has to be useful, not faithful.
- Real-only 45% → co-trained 83% on average. Even off-the-shelf sim data gives +31%.
- The ratio is the key knob, and the best value is extreme (90–99% sim), but not 100%.
- A rough task and camera match is enough. Physics alignment and photorealism add little.
- Behaviour patterns must match. Single-arm sim data teaches a bimanual task the wrong behaviour.
- The same theme as the [data scaling laws]({% post_url 2026-09-20-data-scaling-laws-imitation-learning %}) and [digital cousins]({% post_url 2026-09-22-digital-cousins %}) papers: diversity and coverage beat fidelity.
