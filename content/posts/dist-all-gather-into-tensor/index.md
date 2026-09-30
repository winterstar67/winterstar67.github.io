---
title: "dist.all_gather_into_tensor"
date: 2026-06-13
draft: false
math: true
tags: ["PyTorch", "Distributed training"]
categories: ["Fundamentals"]
description: ""
---

# 1. What's `dist.all_gather_into_tensor`?
`dist.all_gather_into_tensor` is **a collective communication function** in PyTorch.
> Every rank/GPU sends its local tensor, and every rank/GPU receives the concatenation of all ranks' tensors into one output tensor.

# 2. Gradients after applying `dist.all_gather_into_tensor`
Suppose each GPU has its own local gradients after `dist.reduce_scatter_tensor`.

## 2-1. Average gradients (Synchronization, Communication across the GPUs)
- GPU1: 
	- Weights: $[W_1, W_2, W_3]$
	- Averaged Local Gradients: $[\frac{G_{11}+G_{21}+G_{31}}{3}]$
	- Optimizer Local state: $[o_1]$
- GPU2: 
	- Weights: $[W_1, W_2, W_3]$
	- Averaged Local Gradients: $[\frac{G_{12}+G_{22}+G_{32}}{3}]$
	- Optimizer Local state: $[o_2]$
- GPU3:
	- Weights: $[W_1, W_2, W_3]$
	- Averaged Local Gradients: $[\frac{G_{13}+G_{23}+G_{33}}{3}]$
	- Optimizer Local state: $[o_3]$

## 2-2. Update each local weight corresponding to the local gradient from `dist.reduce_scatter_tensor` in each GPU
- GPU1: 
	- Locally Updated Weights: $[W'_1, W_2, W_3]$
- GPU2: 
	- Locally Updated Weights: $[W_1, W'_2, W_3]$
- GPU3:
	- Locally Updated Weights: $[W_1, W_2, W'_3]$

## 2-3. Gather all locally updated weights across GPUs (`dist.all_gather_into_tensor`)
- GPU1: 
	- Updated Weights: $[W'_1, W'_2, W'_3]$
- GPU2: 
	- Updated Weights: $[W'_1, W'_2, W'_3]$
- GPU3:
	- Updated Weights: $[W'_1, W'_2, W'_3]$


## 2-4. Should we use `dist.reduce_scatter_tensor` before calling `dist.all_gather_into_tensor`?
No. It could be a common pattern to use `dist.all_gather_into_tensor` after `dist.reduce_scatter_tensor`, but it's not necessary.
You can think of `dist.all_gather_into_tensor` as a concatenation function across GPUs.
```
# Example
# rank 0 GPU's local_tensor = A  
# rank 1 GPU's local_tensor = B  
# rank 2 GPU's local_tensor = C  
# rank 3 GPU's local_tensor = D

future = dist.all_gather_into_tensor(p, local_tensor, async_op=True).get_future()
```
Here, `p` is an output buffer, and `p == [A, B, C, D]`
The shape of output buffer `p` should be the same as `local_tensor.shape * NUM_GPU`
- More precisely, the shape is `(local_tensor.shape[0]*NUM_GPU, *local_tensor.shape[1:])`

## 2-5. The order of concatenated tensors
The lower rank GPU's local_tensor is placed in the lower index of output_buffer.
In the below code,
```
# Example
# rank 0 GPU's local_tensor = A  
# rank 1 GPU's local_tensor = B  
# rank 2 GPU's local_tensor = C  
# rank 3 GPU's local_tensor = D

future = dist.all_gather_into_tensor(p, local_tensor, async_op=True).get_future()
```
The result is `p == [A, B, C, D] = [rank 0 GPU's local_tensor, rank 1 GPU's local_tensor, ..., rank 3 GPU's local_tensor]`

# 3. How to use `dist.all_gather_into_tensor` function
`future = dist.all_gather_into_tensor(p, local_tensor, async_op=True).get_future()`
- `p`: Output buffer that stores the concatenated local tensors of each GPU. The `p` is overwritten with concatenated tensors.
	- It should be a pre-defined variable.
	- It must have the same shape as `local_tensor.shape * NUM_GPU`
- `local_tensor`: Tensors in each GPU that are concatenated on `p`
- `async_op`: Setting whether this communication operation to run in the background in async way
- `get_future()`: Returns a Future object representing the pending async operation when async_op = True. Calling `.wait()` on it later blocks until the communication finishes.
