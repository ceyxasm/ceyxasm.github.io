---
title: "ChapVidMR: Chapter-based Video Moment Retrieval"
date: 2024-12-01
categories: [Research, Machine Learning]
tags: [computer-vision, nlp, video-understanding, moment-retrieval, multimodal, vlm, icvgip]
---

**ICVGIP 2024** • [DOI](https://dl.acm.org/doi/full/10.1145/3702250.3702282)

---

Video Moment Retrieval (VMR) is the task of finding the specific timestamp in a video that corresponds to a natural language query. You search "the part where they explain the offside rule" and the model returns 4:32–5:10.

Standard VMR has a quiet assumption baked in: each query maps to a single, contiguous moment. That works for short clips but breaks down for long-form video — lectures, documentaries, YouTube deep dives. In a two-hour video, "the part where they explain X" might actually span three separate segments, with digressions in between.

ChapVidMR (Chapter-based Video Moment Retrieval) addresses this.

## The Core Idea

YouTube chapters give us something valuable for free: human-annotated, semantically coherent video segments. Creators add chapters because they naturally divide their content into meaningful units. We exploited this structure as a supervisory signal — rather than trying to localize arbitrary temporal spans, we link natural language queries to sets of chapters.

This reframes VMR as a **multi-label retrieval problem**: a query can and should match *multiple* relevant moments rather than a single one.

## Dataset

No existing VMR dataset supported multi-moment retrieval at this scale, so we built one. Using YouTube-8M as a base and GPT-4o to generate natural language queries aligned to chapter content, we produced a dataset with **12,500+ datapoints** covering diverse video content and query types.

The pipeline combined:
- Video Language Models for visual-semantic alignment
- Audio modality for content that doesn't appear visually (narration, off-screen explanation)
- Prompt engineering to generate varied, naturalistic queries per chapter

## Why Chapters Work

The key insight is that chapters provide **temporal locality with semantic coherence** — something that arbitrary timestamp supervision doesn't. A chapter boundary is a human judgment that the content has meaningfully shifted. Using chapter-aligned retrieval means the model learns to match queries to semantically complete units rather than arbitrary windows.

This also makes evaluation cleaner: instead of computing IoU between predicted and ground-truth timestamps (which is sensitive to annotation disagreement), chapter-level retrieval can be evaluated as a multi-label classification problem with well-defined positive sets.

## Results

The paper was accepted at ICVGIP 2024 (Indian Conference on Vision, Graphics and Image Processing). Full results and methodology at the [DOI link](https://dl.acm.org/doi/full/10.1145/3702250.3702282).
