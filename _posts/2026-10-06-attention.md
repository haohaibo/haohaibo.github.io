---
layout: post
title: attention
date: 2026-10-06 12:00:00
description: Notes and research on attention.
tags: [notes and research]
toc:
  beginning: true
---

Notes and research on attention will be written here.

## Motivation
### 学习与理解attention

Gated DeltaNet是linear attention的一种

linear attention为了获得线性结构和固定 State，放弃了 softmax 的选择能力

$$
\begin{aligned}
\mathrm{Parallel\ training：} &&&\mathbf{O}= (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} &&\in \mathbb{R}^{L\times d}  \\
\mathrm{Iterative\ inference：}&&&\mathbf{o_t} = \sum_{j=1}^t (\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j  &&\in \mathbb{R}^d 
\end{aligned}
$$ 

## How it works

## Notes
