---
title: "Articulate-Anything: Building Articulated 3D Assets with VLM Agents"
date: 2026-08-19 14:30:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [articulate-anything, urdf, robotics, vlm, code-generation, physical-ai, simulation, 3d-assets]
math: true
image:
  path: /assets/img/ran/articulate-anything-hero.png
  alt: "Articulated objects in simulation — chair, window, suitcase, and iron with joint motion traces"
---

> **TL;DR**: Articulated 3D assets (doors that open, drawers that slide, laptops that fold) are essential for robot simulation but painful to build by hand — ~40 minutes per object with dedicated annotation tools. Articulate-Anything automates this by using VLM agents in an actor-critic loop: one agent writes Python code to construct the asset, another renders the result and gives code-level feedback, and they iterate until the output matches the input (text, image, or video). The result: 75% joint-prediction success vs 8–12% for prior methods, and policies trained on generated assets match those trained on human-annotated ones.

> These paper reviews are written more for me and less for others. LLMs have been used in formatting
{: .prompt-tip }

> This post draws from [Articulate-Anything — Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model](https://arxiv.org/abs/2410.13882).
{: .prompt-info }

---

## The Asset Bottleneck

Robot learning in simulation requires articulated 3D objects — cabinets with doors, microwaves with handles, toilets with lids. These are typically represented as **URDFs** (Unified Robot Description Format): a tree of **links** (rigid parts) connected by **joints** (revolute for rotation, prismatic for sliding).

Building these by hand is slow. Even with dedicated tools like RialTo, a single object takes ~40 minutes — scan the real object, segment it into parts, manually specify each joint's type, axis, pivot point, and range of motion. Scale this to the hundreds of objects a diverse simulation environment needs, and asset creation becomes a genuine bottleneck for sim-to-real robotics.

Prior automated approaches (URDFormer, Real2Code) try to predict joint parameters directly — regress coordinates, classify joint types. They achieve 8–12% success rates on the full prediction task. The failure modes are structural: direct regression produces invalid outputs (cyclic structures, repeated links, syntax errors) up to 59% of the time.

---

## The Key Insight: Program Synthesis Beats Coordinate Regression

Instead of predicting numbers, Articulate-Anything generates *code*. A VLM agent writes Python API calls — `place_relative_to(child, parent, "top", clearance=0.01)`, `make_revolute_joint(link, parent, axis=[0,0,1], lower=-90, upper=90, pivot=...)` — that construct the URDF programmatically.

Why is this better?

**Structured output.** Code either executes into a coherent object or throws an error the agent can react to. There's no equivalent of an "invalid output" — the failure mode of direct regression where the predicted structure is internally inconsistent (59% of URDFormer DINO's outputs).

**Composable reasoning.** The VLM can reason about spatial relationships in natural language ("the door hinges on the left side of the cabinet frame") and translate that into precise API calls. It doesn't need to mentally compute 3D coordinates — the API handles collision checking and alignment.

**Iterative refinement.** Code is inspectable. A critic agent can look at the rendered result, compare it to the target, and give *specific code-level feedback* ("try pivoting from the other side", "in our convention left is negative, axis must be [0,0,-1]"). This is impossible with raw coordinate regression — there's no intermediate representation to critique.

---

## The Pipeline: Three Stages

All agents are **Gemini Flash-1.5** instances, each prompted with up to 20 in-context examples. No fine-tuning anywhere.

### Stage 1: Mesh Retrieval

Given the input (text, image, or video), find the closest matching 3D mesh from a library (PartNet-Mobility, ~2,300 objects).

For visual inputs, a CLIP similarity search narrows candidates, then a VLM agent runs a divide-and-conquer tournament — comparing batches of rendered candidates until one winner remains. For text inputs, an LLM densifies the description into per-part specifications, each matched to a library mesh via language-embedding similarity.

### Stage 2: Link Placement

Position each part relative to its parent in the URDF tree. The **actor** VLM writes placement code using the API. The **critic** VLM renders the result, compares it to the input image, and provides:
- A description of what's visually wrong
- A specific suggestion for what to change *in the code*
- A realism rating from 0–10

The loop terminates once the rating exceeds 5 (or the iteration budget runs out — typically 3 iterations).

For video inputs, only the first frame is used for placement — temporal information is reserved for joint prediction.

### Stage 3: Joint Prediction

Extend the placement code with joint-creation API calls. This is where video inputs shine — the actor can observe *how* the object moves (a door swinging, a drawer sliding) to infer joint type, axis, and range.

The **critic** here compares the input video against a **rendered simulation video** of the predicted joint in motion. It attributes errors to specific categories in severity order: wrong joint type (most egregious) → wrong axis → wrong origin → wrong limits. The feedback is again code-level: "the axis should be [0,0,-1] not [0,0,1]."

---

## The Actor-Critic Loop in Action

A worked example from the paper — predicting a door hinge:

| Iteration | Rating | Critic Feedback |
|-----------|--------|----------------|
| 1 | 2/10 | "Try pivoting from the other side" |
| 2 | 3/10 | "Sign convention issue — axis must be [0,0,-1] not [0,0,1]" |
| 3 | 10/10 | No further suggestions |

This loop adds +5.8% to link placement success and +2.4% to joint prediction over single-pass generation. Most gains come by iteration 2 — diminishing returns after that.

The critic agrees with ground truth ~93% of the time. The dominant error mode is false positives — the critic wrongly accepts a subtly incorrect result. Subtle geometric mismatches are hard to catch visually, even for a VLM.

---

## How Much Do In-Context Examples Matter?

A lot, especially for joint prediction:

| # Examples | Link Placement | Joint Prediction |
|-----------|---------------|-----------------|
| 0 | 58% | 5% |
| 5 | 90% | 18% |
| 20 | ~96% | 78% |

Joint prediction is dramatically more example-hungry than link placement. Specifying axes, pivot points, and sign conventions requires precise geometric reasoning that benefits heavily from worked examples. Link placement is a simpler spatial arrangement task that the VLM handles reasonably even near-zero-shot.

---

## Results

### vs Baselines

| Method | Success Rate | Notes |
|--------|-------------|-------|
| **Articulate-Anything** | **75%** | No category-specific training |
| URDFormer (oracle boxes) | 8.7% | Given ground-truth bounding boxes |
| URDFormer (DINO) | 1.3% | Detected bounding boxes |
| Real2Code (oracle) | 12.2% | Given oracle RGB-D + segmentation |

Even when tested on the same restricted input modality as each baseline (text-only vs Real2Code, image-only vs URDFormer), Articulate-Anything still wins by 2–3x. The gap isn't just about having better inputs — the program-synthesis + actor-critic pipeline itself is doing real work.

### Joint Type Accuracy

Articulate-Anything's joint-type error rate is ~2% — roughly 20–25x lower than either baseline. Getting joint type wrong (predicting revolute when it should be prismatic, or vice versa) is the most fundamentally broken failure mode, and the baselines get it wrong 40–55% of the time.

### Failure Breakdown

| Failure Mode | Articulate-Anything | URDFormer DINO | Real2Code |
|-------------|--------------------|----|---|
| Success | 75.0% | 8.7% | 12.2% |
| Wrong joint type | 1.7% | 18.8% | 43.9% |
| Invalid output | 0% | **59.1%** | 17.5% |
| Link placement error | 14.7% | — | — |

The "invalid output" row is the clearest argument for program synthesis over coordinate regression. URDFormer produces structurally invalid predictions (cyclic structures, repeated links) in 59% of cases. This failure mode simply doesn't exist when the output is executable code.

### VLM Backbone Robustness

| Backbone | Success Rate |
|----------|-------------|
| GPT-4o | 70% |
| Gemini Flash-1.5 | 78% |
| Claude-3.5 Sonnet | 86% |

All three well above baselines — the result isn't an artifact of one specific model.

---

## Does It Matter for Robot Learning?

The practical test: train manipulation policies via PPO in simulation on Articulate-Anything's generated assets vs human-annotated assets (using RialTo's dedicated scanning/annotation workflow). Four fine-grained manipulation tasks, 3 seeds each, 2 million environment steps.

**Both reach 100% real-world success rate** when deployed on a physical Franka Panda arm. The generated assets are functionally equivalent for downstream RL training, while saving ~40 minutes of human labor per object.

---

## Where It Struggles

Per-category success ranges from 0% (scissors, fans, pliers, eyeglasses — small parts, exotic joint geometries) to 98% (remotes). The main difficulty drivers are small object parts and irregular movement patterns. The mesh-retrieval constraint also bounds output diversity to what's in the library — a mesh-generation extension (using Rodin for single-image 3D generation) shows promise but is preliminary.

---

## Key Takeaways

- **Program synthesis beats coordinate regression** for articulated asset generation — 75% vs 8–12%, with zero structurally invalid outputs vs 17–59% for baselines.
- The **actor-critic loop** (propose code → render → VLM critiques with code-level feedback → iterate) is a reusable design pattern for any "generate structured artifact, verify via rendering" pipeline.
- **Video inputs** are critical for joint prediction — they provide motion cues that images and text can't: joint type from how the part moves, axis from the direction of motion, range from the extent.
- In-context examples matter far more for geometric reasoning (joint prediction: 5% → 78% across 0 to 20 examples) than for spatial arrangement (link placement: 58% → 96%).
- Generated assets are **functionally equivalent** to human-annotated ones for downstream robot policy training — same 100% real-world success, ~40 minutes saved per object.
