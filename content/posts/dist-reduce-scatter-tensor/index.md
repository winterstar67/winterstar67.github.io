---
title: "dist.reduce_scatter_tensor"
date: 2026-06-13
draft: false
math: true
tags: ["PyTorch", "Distributed training"]
categories: ["Fundamentals"]
description: ""
---

# 1. What's `dist.reduce_scatter_tensor`?
`dist.reduce_scatter_tensor` is **a collective communication function** in PyTorch.
> **reduce first, then scatter the reduced result across ranks.**

# 2. Gradients after applying `dist.reduce_scatter_tensor`
Suppose each GPU computes its own local gradients via forward and backward, then reduces and scatters them.

## 2-1. Forward and backward
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

## 2-2. Reduce(average) gradients and scatter (`dist.reduce_scatter_tensor`)
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

# 3. The order of allocating split tensor
The split order is also rank-based:

```
rank 0 GPU gets first shard
rank 1 GPU gets second shard
rank 2 GPU gets third shard
rank 3 GPU gets fourth shard
```

# 4. How to use `dist.reduce_scatter_tensor` function
`future = dist.reduce_scatter_tensor(grad_slice, p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()`
- `grad_slice`: Output tensor that holds that rank's sharded tensor
- `p.grad`: It's an input tensor. It's the full size of the tensor in each GPU. (In nanochat, it's a gradient in each GPU from different data)
- `op`: Reduction operation, which is combining multiple values into one value
	> take all GPUs' `p.grad` tensors and average them.
- `async_op`: Setting whether this communication operation to run in the background in async way
- `get_future()`: Returns a Future object representing the pending async operation when async_op = True. Calling `.wait()` on it later blocks until the communication finishes.

`dist.reduce_scatter_tensor` requires `input size = chunk_size * NUM_GPU` exactly.
- If the input size is not divisible by world_size, we pad it with zeros to meet this requirement. The padding is ONLY for the communication across the GPUs. It's not for other operations.
