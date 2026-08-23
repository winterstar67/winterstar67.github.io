---
title: "ZeRO-2"
date: 2026-06-12
draft: false
math: true
tags: ["Distributed training"]
categories: ["Fundamentals"]
description: ""
---

# 1. What is ZeRO-2?
**ZeRO-2** is one of the ways to do distributed training.
Each GPU has
- The full set of model parameters
- Mini batch shard of data
- A shard of the gradients
- Part of the optimizer states, such as part of the momentum in Adam

# 2. The flow of ZeRO-2
Suppose each GPU loads exactly the same weights, but uses different data. Here's a simple example
## 2-1. Initial settings (Loading Data and Model parameter)
- GPU1: 
	- Weights: $[W_1, W_2, W_3]$
	- Data Batch: $[DB_{11}, DB_{12}, DB_{13}, DB_{14}]$ 
- GPU2: 
	- Weights: $[W_1, W_2, W_3]$
	- Data Batch: $[DB_{21}, DB_{22}, DB_{23}, DB_{24}]$
- GPU3:
	- Weights: $[W_1, W_2, W_3]$
	- Data Batch: $[DB_{31}, DB_{32}, DB_{33}, DB_{34}]$ 
- $DB_{ij}$ : j-th data in i-th GPU

## 2-2. Forward and backward
- GPU1: 
	- Weights: $[W_1, W_2, W_3]$
	- Forward activations: $[F_{11}, F_{12}, F_{13}, F_{14}]$ 
	- Backward Gradients: $[G_{11}, G_{12}, G_{13}]$ 
- GPU2: 
	- Weights: $[W_1, W_2, W_3]$
	- Forward activations: $[F_{21}, F_{22}, F_{23}, F_{24}]$ 
	- Backward Gradients: $[G_{21}, G_{22}, G_{23}]$ 
- GPU3:
	- Weights: $[W_1, W_2, W_3]$
	- Forward activations: $[F_{31}, F_{32}, F_{33}, F_{34}]$ 
	- Backward Gradients: $[G_{31}, G_{32}, G_{33}]$ 
- $F_{ij}$: Forward activations of j-th data in i-th GPU
- $G_{ij}$: Backward gradients of j-th weight in i-th GPU

## 2-3. Average gradients (`dist.reduce_scatter_tensor`)
- GPU1: 
	- Weights: $[W_1, W_2, W_3]$
	- Averaged Local Gradients: $[\frac{G_{11}+G_{21}+G_{31}}{3}]$
	- Optimizer Local State: $[o_1]$
- GPU2: 
	- Weights: $[W_1, W_2, W_3]$
	- Averaged Local Gradients: $[\frac{G_{12}+G_{22}+G_{32}}{3}]$
	- Optimizer Local State: $[o_2]$
- GPU3:
	- Weights: $[W_1, W_2, W_3]$
	- Averaged Local Gradients: $[\frac{G_{13}+G_{23}+G_{33}}{3}]$
	- Optimizer Local State: $[o_3]$
- **Now, each GPU doesn't hold gradients of every parameter. Only part of the averaged gradients is stored in each GPU**
	- This is the part that makes ZeRO-2 a memory efficient method
- The optimizer, such as AdamW or Muon, operates on the part of the grads in each GPU

## 2-4. Weight update
Part of the weights is updated in each GPU based on the update value after the optimizer step.
- GPU1: 
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W_{2}, W_{3}]$ - $W_1$ is updated
- GPU2: 
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W_{1}, W'_{2}, W_{3}]$ - $W_2$ is updated
- GPU3:
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W_{1}, W_{2}, W'_{3}]$ - $W_3$ is updated

## 2-5. All gather
After updating partial weights in each GPU, we gather each updated weight so that every GPU has all of the updated weights
- GPU1: 
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W'_{2}, W'_{3}]$
- GPU2: 
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W'_{2}, W'_{3}]$
- GPU3:
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W'_{2}, W'_{3}]$

# 3. Comparison with DDP
It has more steps to complete the synchronization across the GPUs
## 3-1. Steps of DDP
```
1. Load the full set of model parameters
2. Calculate all forward activations/backward gradients
3. Average all gradients in all GPUs
4. Run the optimizer algorithm and update all parameters in each GPU
```
It's faster than ZeRO-2 but takes more memory

## 3-2. Steps of ZeRO-2
```
1. Load the full set of model parameters
2. Calculate all forward activations/backward gradients
3. Do reduce scatter and each GPU gets part of the averaged gradients and frees the other unallocated gradients
4. Run the optimizer algorithm and update the part of the parameters in each GPU
5. Gather all results so that every GPU has the same updated weights
```
It's slower than DDP but more memory efficient

# 4. ZeRO-2 Code in nanochat
`_reduce_muon`: Average the gradients and split each averaged gradient into GPUs
`_compute_muon`: Based on part of the grads in GPUs, run the optimizer algorithm and update part of the parameters that correspond to each grad.
`_finish_gathers`: copies it back to the original model parameters (The gather above operates on a separate (copied) memory buffer, not the original parameter tensor)
