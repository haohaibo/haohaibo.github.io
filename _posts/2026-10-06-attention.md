---
layout: post
title: Attention
date: 2026-10-06 12:00:00
description: Notes and research on attention.
tags: [notes and research]
toc:
  beginning: true
---

Notes and research on attention will be written here.

## Motivation
### 学习与理解attention
Full attention

Vanilla full attention的计算公式如下

$$
\begin{aligned}
\mathrm{Parallel \ training:} &&& \mathbf{O} = \mathrm{softmax}\left(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}\right) \mathbf{V} && \in \mathbb{R}^{L \times d} \\

\mathrm{Iterative \ inference:} &&& \mathbf{o}_t = \sum_{j=1}^t \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)} {\sum_{l=1}^t \exp(\mathbf{q}_t^\top \mathbf{k}_l)} \mathbf{v}_j && \in \mathbb{R}^d

\end{aligned}
$$

Full attention去掉softmax得到linear attention

Gated DeltaNet是linear attention的一种

linear attention为了获得线性结构和固定 State，放弃了 softmax 的选择能力

$$
\begin{aligned}
\mathrm{Parallel\ training:} &&&\mathbf{O}= (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} &&\in \mathbb{R}^{L\times d}  \\
\mathrm{Iterative\ inference:} &&&\mathbf{o}_t = \sum_{j=1}^t (\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j  &&\in \mathbb{R}^d 
\end{aligned}
$$ 

## How it works

### 理解Attention是在做什么
这里有一个很好的视频解释：[Attention in transformers, visually explained | 3Blue1Brown](https://youtu.be/eMlx5fFNoYc)

## Notes
