---
title: "Kernel Investigation"
date: 2026-08-14
draft: false
math: true
tags: ["Analysis", "Kernel Selection"]
categories: ["Experiments"]
description: ""
---

![Kernel Investigation](kernel-investigation.png)

> **Update (2026-09-29):** An earlier version of this post said that unaligned shapes prevent Tensor Core usage. That only holds for cuBLAS < 11.0. In cuBLAS 11.0 or later, alignment affects kernel efficiency, not whether Tensor Cores are used.

The implementation and experiment code is on [GitHub](https://github.com/winterstar67/AI-Experiments/tree/main/Kernel%20investigation).

Related posts:
- {{< wikilink "factors-to-consider-in-cuda-and-gpu" >}}

# 1. Motivation
While I was writing the code to evaluate the model on HellaSwag dataset, I saw that a more efficient kernel can be selected by weight shape (vocab size)
- Megatron used vocab size padding
- nanochat used vocab size padding

So, based on this observation, I thought that **maybe the shape of input data can affect the kernel selection so inference speed could become faster too**
- [FP16 requires a multiple of 8 elements for efficient operation](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)
- [cuBLAS uses column-major](https://docs.nvidia.com/cuda/cublas/index.html)

# 2. Hypothesis
If I set the batch size and token length with padding appropriately (a multiple of 8), CUDA would select the faster kernel.

# 3. Settings
- I used HellaSwag data, so the batch size is a multiple of 4 (`B = 4*n`)
- **The versions of CUDA and cuBLAS are 12.8 and 12.8.4.1, respectively**

After running the code, I analyzed the kernel of the linear layer that embeds the activation to the vocab size dimension in the last layer by uploading the profiler trace file on Perfetto UI

# 4. Test cases
I tested a total of 8 cases
- FP32 vs. FP16
- Vocab size padded vs. not padded
- Batch size = 4×26 with token length ≡ 0 (mod 16), vs. batch size = 4×25 with odd token length.

The images below indicate the kernel of the linear layer embedding to the vocab size dimension in the last layer.

## 4-1. float32 case - No Tensor Core in T4 GPU
### 4-1-1. vocab padding, float32, BatchSize25, Token size padding odd case
![vocab padding float32 BatchSize25 Padding odd](vocab-padding-float32-batchsize25-padding-odd.png)
- Kernel: volta_sgemm_128x128_tn
- Duration: 253$ms$ 733$\mu s$
### 4-1-2. vocab padding, float32, BatchSize26, Token size padding even
![vocab padding float32 BatchSize26 Padding even](vocab-padding-float32-batchsize26-padding-even.png)
- Kernel: volta_sgemm_128x128_tn
- Duration: 278$ms$ 575$\mu s$
### 4-1-3. No vocab padding, float32, BatchSize25, Token size padding odd
![No vocab padding float32 BatchSize25 Padding odd](no-vocab-padding-float32-batchsize25-padding-odd.png)
- Kernel: volta_sgemm_128x128_tn
- Duration: 263$ms$ 286$\mu s$

### 4-1-4. No vocab padding, float32, BatchSize26, Token size padding even
![No vocab padding float32 BatchSize26 Padding even](no-vocab-padding-float32-batchsize26-padding-even.png)
- Kernel: volta_sgemm_128x128_tn
- Duration: 291$ms$ 657$\mu s$


## 4-2. float16 case - Tensor Core supported in T4 GPU
### 4-2-1. vocab padding, float16, BatchSize25, Token size padding odd
![vocab padding float16 BatchSize25 Padding odd](vocab-padding-float16-batchsize25-padding-odd.png)
- Kernel: turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn
- Duration: 42$ms$ 613$\mu s$
### 4-2-2. vocab padding, float16, BatchSize26, Token size padding even
![vocab padding float16 BatchSize26 Padding even](vocab-padding-float16-batchsize26-padding-even.png)
- Kernel: turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn
- Duration: 47$ms$ 378$\mu s$
### 4-2-3. No vocab padding, float16, BatchSize25, Token size padding odd
![No vocab padding float16 BatchSize25 Padding odd](no-vocab-padding-float16-batchsize25-padding-odd.png)
- Kernel: void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_128x256_tn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_128x256_tn_align1::Params)
- Duration: 101$ms$ 291$\mu s$
### 4-2-4. No vocab padding, float16, BatchSize26, Token size padding even
![No vocab padding float16 BatchSize26 Padding even](no-vocab-padding-float16-batchsize26-padding-even.png)
- Kernel: void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_tn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_tn_align1::Params)
- Duration: 111$ms$ 186$\mu s$

## 4-3. Kernel names
- `sgemm`: Single precision GEMM, FP32 input and output
- `128x128`: threadblock tile, each thread block computes a `128x128` output tile
- `tn`: first matrix Transposed, second Not transposed
- `cutlass_75`: A kernel for SM75(Turing) made with CUTLASS
- `tensorop`: Tensor Core is used
- `align1`: Loading only one element (2 bytes in FP16) at once
- `ldg8`: Loading eight elements (16 bytes in FP16) at once (not officially documented)

# 5. Result
## 5-1. FP32 case
Four FP32 cases show the same kernels. Every vocab size, batch size, and token size didn't affect the kernel selection.

## 5-2. FP16 case
Vocab size padding selects a different kernel, which is about 2.4x faster than the non-padded case.
But, the batch size and token size didn't lead to the aligned kernel as I hypothesized.

# 6. Analysis
Why didn't the batch size and token size affect the selection of the aligned kernel in FP16? I'll explain it in this section.
- First of all, the data shape `(B,T,K)` is treated as `(B*T,K)` in the linear layer. So B and T should not be considered separately, but as a single value, B$*$T.
- B$*$T = 13700 (a multiple of 4 but not 8) vs. 14976 (a multiple of 8) were tested and both selected the same type of kernel (though the tilings are different in the non-padded case). So whether B$*$T is a multiple of 8 or not did not matter.

The reason is that B$*$T is not a leading dimension, while N is. So B and T don't affect the selection of the aligned kernel.
> By the way, [Tensor Core for FP32(TF32) is not supported in T4 GPU](https://developer.nvidia.com/blog/?p=11872). Only the INT8, INT4, and FP16 are supported. FP32 is used for accumulation in mixed precision in Tensor Core.

## 6-1. Explanation of leading dimension
Suppose the shapes of variables in the last layer, `y = X @ W.T`, are the following (`B`: batch size, `T`: token length, `K`: embed dim, `N`: vocab size):
- `X` shape: `(B, T, K)`
- `W` shape: `(N, K)`
- `y` shape: `(B, T, N)`

Because a more efficient kernel operates on data in units of 16 bytes, we should allocate the data to be a multiple of 8 in the FP16 case.
Here, the dimension axis is important. We have an input X with shape `(B,T,K)`.
When that input goes through the linear layer, the layer treats it as `(B*T, K)` form.
The weight of the linear layer, `W`, is `(N, K)` and when the linear layer is run, the operation is `y = X @ W.T`.
`y = X @ W.T` is a PyTorch representation, and PyTorch aligns the data in row-major order, which means the data in memory is filled one row at a time, completing a full row before moving to the next.
- If there's no manipulation of stride and `(K, N)` shape data has stride `(N, 1)`, it means that we need to jump `N * dtype's bytes` in memory to reach the next row.

Unlike PyTorch, the data alignment in cuBLAS is different. cuBLAS stores data in a column-major way.
So, keeping in mind that cuBLAS reads the data in a column-major way, if we investigate it,
- The weight `W` in PyTorch is stored as `W.T` whose shape is `(K, N)` in cuBLAS. The stride in cuBLAS is `(1, K)`. This is `lda`.
- The input `X` in PyTorch is stored in `X.T` whose shape is `(K, B*T)`. The stride in cuBLAS is `(1, K)`. This is `ldb`.
- `y=X @ W.T` in PyTorch becomes `y.T = W @ X.T` whose shape is `(N, B*T)` in cuBLAS. So the stride in cuBLAS is `(1, N)`. This is `ldc`.

![PyTorch vs cuBLAS shapes, strides, and memory layout](cublas.png)

The factors that affect the efficiency of an operation are `K` and `N` which are embedding dimension and vocab size respectively.
- **In PyTorch, if we transpose W, then the stride of W.T is also transposed, which means the row-major becomes column-major.**
- **In cuBLAS, even if we transpose W.T into W, still the stride or memory access is done in column-major order of W.T**

The reason that vocab size decides whether the aligned kernel is selected, while batch size and token size don't, is that there is `N` in the stride but there are no B and T terms in the stride.

But, it doesn't mean that the batch size doesn't affect kernel selection at all. It can still change the tile shape of the selected kernel.
- For example, in the two non-padded FP16 cases, the results showed different kernels
	- `...cutlass_75_tensorop_f16_s1688gemm_f16_128x256_tn_align1`
	- `...cutlass_75_tensorop_f16_s1688gemm_f16_256x128_tn_align1`
