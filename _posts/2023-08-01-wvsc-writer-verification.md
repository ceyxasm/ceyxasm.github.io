---
title: "WVSC-I: Writer Verification at NCVPRIPG 2023"
date: 2023-08-01
categories: [Research, Computer Vision]
tags: [computer-vision, handwriting, writer-verification, challenge, pattern-recognition, biometrics]
---

**NCVPRIPG 2023** • [DOI](https://link.springer.com/chapter/10.1007/978-981-97-5212-6_15)

---

Given two images of handwritten text, were they written by the same person?

This is the writer verification problem — a form of biometric identification without any dynamic information. No pen pressure sensors, no stroke timing, no stylus trajectory. Just the ink on the page.

WVSC-I (Writer Verification Summer Challenge, Edition I) was a competition I organized at NCVPRIPG 2023 (National Conference on Computer Vision, Pattern Recognition, Image Processing and Graphics). This post covers what the challenge was, why it's hard, and what we learned from running it.

## Why It's Hard

Handwriting is variable even within a single writer. Fatigue, speed, surface texture, pen type — all of these cause genuine intra-writer variation. The model has to distinguish *intra-writer variation* (same writer, looks different) from *inter-writer similarity* (different writers, looks similar), without any signal beyond the static image.

The specific challenge was **offline handwritten Hindi text** — an additional layer of complexity over Latin-script writer verification, because Hindi has more complex conjunct characters, ligatures, and stroke patterns, and existing datasets for offline Hindi handwriting verification are limited.

The application context: fraud detection. Signature forgery, document authentication, disputed will cases — writer verification is a real forensic problem.

## The Challenge Structure

Running a challenge is a different kind of research work. The pipeline:

1. **Data creation and cleaning** — sourcing handwritten Hindi samples, ensuring quality and balance across writers
2. **Baseline development** — establishing a minimum performance threshold so participants have something to beat
3. **Platform and evaluation** — interactive website, submission pipeline, leaderboard
4. **Community** — QnA sessions with participants, clarifying the task formulation as edge cases emerged

The proceedings were later prepared for NCVPRIPG 2024.

## What the Task Requires

Effective approaches to writer verification typically need to learn **style embeddings** — representations that compress a writer's individual characteristics (stroke thickness, slant, letter spacing, pressure distribution as revealed by ink density) into a vector, then measure distance between vectors from the two samples.

The challenge is that these embeddings must be robust to content variation — the same writer writing different words — while remaining sensitive to style variation between writers. That's a metric learning problem with some specific difficulties in the Hindi domain.

## Takeaway

Organizing rather than just competing in a challenge teaches you different things: the ambiguities in your own task definition, the failure modes of your evaluation metric, and the ways participants interpret your problem that you didn't anticipate. WVSC-I was a useful exercise in all three.

Full report in the [proceedings](https://link.springer.com/chapter/10.1007/978-981-97-5212-6_15).
