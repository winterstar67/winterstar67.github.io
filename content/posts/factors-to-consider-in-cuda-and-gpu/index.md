---
title: "Factors to Consider in CUDA and GPU"
date: 2026-08-15
draft: false
math: true
tags: ["CUDA", "GPU", "Tensor Core"]
categories: ["Fundamentals"]
description: ""
---

**Note: This post is just accumulation of what I investigated**

Setting input shape and weight shape to be a multiple of 16 is not the only thing to consider when considering an efficient kernel

# 1. GPU
There are many GPUs such as T4, A100, B100, etc. Each GPU has its own structure.

## 1-1. Tensor Core
**The Tensor Core** is a hardware unit that processes the matrix multiplication in a faster way than a CUDA core.
When several rules are satisfied, the Tensor Core is activated. In {{< wikilink "kernel-investigation" >}}, the Tensor Core runs about 2x faster than the normal one.

The Tensor Core can affect the following:
1. Forward activations
2. Activation gradients
3. Weight gradients
- Reference: [Figure 7](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#checklist)

### 1-1-1. The rules to utilize the Tensor Core
```
((op_A == N ? m : k) * AtypeSize) % 16 == 0
((op_B == N ? k : n) * BtypeSize) % 16 == 0
(m * CtypeSize) % 16 == 0

(lda * AtypeSize) % 16 == 0
(ldb * BtypeSize) % 16 == 0
(ldc * CtypeSize) % 16 == 0

A pointer % 16 == 0
B pointer % 16 == 0
C pointer % 16 == 0
```
- Here, the `lda`, `ldb`, and `ldc` above can be considered as the strides of A, B, and C in $C = A@B$
- Reference: [2.1.11. Tensor Core Usage](https://docs.nvidia.com/cuda/cublas/index.html#tensor-core-usage)

On the A100, a multiple of 128 is also supported. The other requirements could be applied or added. So, figure out which hardware (GPU) is being used first and search what Tensor Core is supported on that GPU.

### 1-1-2. Tensor Core Requirements in GPU
If we want to utilize the Tensor Core for a certain data type, the GPU must support that Tensor Core of that dtype.
Even if CUDA has an implementation that utilizes the Tensor Core, if the GPU doesn't support the Tensor Core, then we can't utilize the Tensor Core.

# 2. CUDA
This is a software platform installed in a runtime environment. There are various versions of CUDA and it's not dependent on the GPU. Even if the GPU supports a certain dtype, if the applied CUDA version doesn't have the kernel implementation to utilize that Tensor Core, then there's no way to run the efficient kernel using the Tensor Core.
So, both **GPU** and **CUDA** must support the implementation and architecture for that dtype.
- The possible option of kernel selection is roughly the intersection of GPU and CUDA

# 3. Efficiency difference
By the selection of input shape or weight shape, the kernel selection and memory efficiency could be different

## 3-1. Terminologies
- CTA: It's the same as Thread Block.

## 3-2. Tile quantization efficiency
**Tile quantization** is a phenomenon where, given an `n x n` tile (Suppose `n=128`, each 128 means one data element in a matrix), the last tile takes too little data.
It's not necessarily related to the kernel selection.
The efficiency of the tile quantization is decided by the number of elements, not bytes.

If the batch size is 129, then we need two 128 tiles.
```
batch 129

CTA tile #1:
[████████████████] 128 / 128

CTA tile #2:
[█...............]   1 / 128
```
Here, the second tile is almost empty, which is a waste of computational resources.
If we set the batch size to 256, then
```
batch 256

CTA tile #1:
[████████████████] 128 / 128

CTA tile #2:
[████████████████] 128 / 128
```
Two tiles are full of data.
Suppose the total amount of data is 1024. Then the `batch 129` case runs the operation $\lceil{\frac{1024}{129}}\rceil = 8$ times while the `batch 256` case runs the operation $\lceil{\frac{1024}{256}}\rceil = 4$ times.
But naturally, batch 256 processes more data at once than batch 129.
Suppose our memory can hold a batch size of 129 at most. Then, 129 could be a better selection than 128 or other aligned data settings, but it seems there's no big difference.

The tile size is described in the kernel name in the Perfetto UI

## 3-3. Wave quantization efficiency
**Wave quantization** is a phenomenon where there are many CTAs but the last SMs are idle during the last wave. This wastes SMs.
It's not necessarily related to the kernel selection.
The efficiency of the wave quantization is decided by the number of elements, not bytes.
Suppose we have 81 CTAs and the SMs can handle 40 CTAs at once, then
```
1st wave: 40 CTAs
2nd wave: 40 CTAs
3rd wave:  1 CTA
```
Now, at the last wave, the SM state would be
```
SM 0  → CTA 81 execution
SM 1  → idle
SM 2  → idle
...
SM 39 → idle
```

The number of SMs is a spec of the GPU.
The number of CTAs (thread blocks) is described in the block part of the kernel description in the Perfetto UI.
```
block
[0] 256
[1] 1
[2] 1
```
Because 1 warp has 32 threads, block size = $256*1*1=256$ means there are 8 warps = 256 threads.

## 3-4. Kernel selection efficiency by the Tensor Core
It's related to the kernel selection. It is decided by the total bytes, which is `total bytes = the number of elements * dtype bytes`.
The way to utilize the Tensor Core is described above
