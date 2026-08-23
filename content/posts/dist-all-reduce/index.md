---
title: "dist.all_reduce"
date: 2026-06-12
draft: false
math: true
tags: ["PyTorch", "Distributed training"]
categories: ["Fundamentals"]
description: ""
---

# 1. What's `dist.all_reduce`?
`dist.all_reduce` is a **collective communication function** in PyTorch distributed training.
> Every process/GPU sends its tensor, PyTorch reduces them together, and then every process/GPU receives the same reduced result.

# 2. Gradients after applying `dist.all_reduce`
Suppose we average the gradients 
## 2-1. Setup
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

## 2-3. Average gradients (Synchronization, Communication across the GPUs)
- GPU1: 
	- Weights: $[W_1, W_2, W_3]$
	- Backward Gradients: $[G_{11}, G_{12}, G_{13}]$ 
	- Averaged Backward Gradients: $[\frac{G_{11}+G_{21}+G_{31}}{3}, \frac{G_{12}+G_{22}+G_{32}}{3}, \frac{G_{13}+G_{23}+G_{33}}{3}]$
- GPU2: 
	- Weights: $[W_1, W_2, W_3]$
	- Backward Gradients: $[G_{21}, G_{22}, G_{23}]$ 
	- Averaged Backward Gradients: $[\frac{G_{11}+G_{21}+G_{31}}{3}, \frac{G_{12}+G_{22}+G_{32}}{3}, \frac{G_{13}+G_{23}+G_{33}}{3}]$
- GPU3:
	- Weights: $[W_1, W_2, W_3]$
	- Backward Gradients: $[G_{31}, G_{32}, G_{33}]$ 
	- Averaged Backward Gradients: $[\frac{G_{11}+G_{21}+G_{31}}{3}, \frac{G_{12}+G_{22}+G_{32}}{3}, \frac{G_{13}+G_{23}+G_{33}}{3}]$

# 3. How to use `dist.all_reduce` function
`future = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()`
- `p.grad`: Gradients in each GPU that would be communicated with `op` operation
- `op`: Reduction operation, which is combining multiple values into one value
	> take all GPUs' `p.grad` tensors and average them.
- `async_op`: Setting whether this communication operation to run in the background in async way
- `get_future()`: Returns a Future object representing the pending async operation when `async_op = True`. Calling .wait() on it later blocks until the communication finishes.
