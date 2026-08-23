---
title: "Thread, Warp, and Block in GPU execution"
date: 2026-05-09
draft: false
math: true
tags: ["Background"]
categories: ["Fundamentals"]
description: ""
---

# 0. Background

## 0-1. CUDA
CUDA is a platform developed by NVIDIA. It extends C/C++ with GPU-specific syntax, allowing the code to run on the GPU

## 0-2. GPU
Compared to CPU, GPU contains lots of cores while the ability of each core is simpler than a CPU's core. It's suitable for large and simple operations

## 0-3. Kernel
Even one operation can have multiple implementations that all implementations make the same results.
- In this post, matrix multiplication(matmul) is used as an example. Even the matrix multiplication has many implementations in CUDA.
- To distinguish a function(matrix multiplication) itself from the various routes to implement it, each of the various implementations is called `kernel`. 

The example of the same operation but various implementations: Mean in statistics
When one data point is added in every step, how would you recalculate the mean of all data?
1. Mean definition: $\frac{x_1 + x_2 + ... + x_n}{n}$ - duplicated data use
2. Incremental way: $u_k = u_{k+1} + \frac{x_k - u_{k-1}}{k}$ - More efficient
Both make the same result but the calculation is quite different. And the efficiency is also different

## 0-4. The hierarchy of CUDA operation
```
  Grid
    └── Block 0, Block 1, Block 2 ...
            └── Warp 0, Warp 1 ...
                     └── Thread 0~31 (Execute in parallel)
```

# 1. Thread
The smallest unit of operation
One thread takes one element.

Question 1: What's the one element in $C_{ij}$ = matmul$(A_{ij}, B_{ij})$?
Answer: 
- A element that one thread takes is $C[i,j]$.
- One thread that takes $C[i,j]$ operates $\sum_x{A[i,x]*B[x,j]}$
	- By setting one thread to take one matmul operation, every thread can do exactly the same operation of matmul

```
  A: (3×3), B: (3×3), C: (3×3)
  C[i][j] = i-th row in matrix A · j-th column in matrix B (dot product)

  CUDA: One C element per one thread

  thread(0,0) → C[0][0] calculation
  thread(0,1) → C[0][1] calculation
  thread(0,2) → C[0][2] calculation
  thread(1,0) → C[1][0] calculation
  ...
  thread(2,2) → C[2][2] calculation ← 9 threads at once

  Each thread:
  thread(i, j):
    result = 0
    result += A[i][0] * B[0][j]
    result += A[i][1] * B[1][j]
    result += A[i][2] * B[2][j]
    C[i][j] = result

# Example By Claude Code
```

# 2. Warp
It's the group of threads. 32 threads become one warp
Why is there the terminology or concept "warp"? Is it just a group of 32 threads? What about 20 threads? Why is the 32 special?

## 2-1. Concepts to understand warp
- 1 Instruction fetch
- 1 Decode
- 1 PC
- 1 active mask (32 bits)
- 32 lane

Instruction fetch:
- Reading the next instruction from instruction memory
- It's like a commander that order which operation to execute.
- If there are several instruction fetches, we can order several operations in parallel for several threads. But GPU has only one instruction fetch per warp.

Decode:
- Converting human readable instruction into machine readable instruction

PC:
- Which instruction to execute, the order(number)

Lane:
- one execution path from a warp to a thread
- in warp = 32 threads situation:

```
Warp
  ├── lane 0
  ├── lane 1
  ├── lane 2
  ...
  └── lane 31
```

- Each lane corresponds to one thread's data.

```
In warp execution:

one shared PC  
→ one instruction fetch  
→ one decode  
→ broadcast to 32 lanes
```

```
# ChatGPT

In NVIDIA CUDA:

- 1 warp = 32 threads

Those 32 threads execute:

- same instruction
- same cycle
- same program counter

This is called:

- SIMT = Single Instruction Multiple Threads
```

## 2-2. Full warp flow example

Suppose warp PC says:

```
PC = instruction #20
```


Hardware does:
### 2-2-1: Fetch

```
Fetch instruction #20
```

Suppose instruction is:

```
ADD R1, R2, R3
```

meaning:

```
R1 = R2 + R3
```

### 2-2-2: Decode
Hardware understands:

```
operation = ADD
inputs = R2, R3
output = R1
```

### 2-2-3: Execute across lanes
Warp executes simultaneously:

```
lane0: R1 = R2 + R3
lane1: R1 = R2 + R3
lane2: R1 = R2 + R3...
```

But each lane has different register values.

Why this is efficient

Without warp sharing:

```
32 fetches
32 decodes
32 PCs
```

With warp:

```
1 fetch
1 decode
1 PC
32 data lanes
```

Massive hardware savings. That is one of the fundamental reasons GPUs achieve huge throughput.

## 2-3. All active threads in warp execute same instruction
Example:

```
ADD
```

Then:

```
lane0: add
lane1: add
lane2: add...
```

Same instruction.

## 2-4. Warp size is divided with 32
If a thread block contains a number of threads not evenly divisible by the warp size, the SM creates a partially filled final warp that still consumes the full warp's resources. For example, if a thread block has 100 threads and the warp size is 32, the SM creates:

- 3 full warps of 32 threads each (96 threads total)
    
- 1 partial warp with only 4 active threads but still occupying a full warp's worth of resources (32 thread slots)
    
The SM effectively disables the unused thread slots in partial warps, but these slots still consume hardware resources. For this reason, developers generally should make thread block sizes a multiple of the warp size to optimize resource usage.

citation: https://docs.modular.com/glossary/gpu/warp/

# 3. Block
It's the group of threads. Usually, one block can hold up to 1024 threads.
Again, why is the block concept special?
- Claude said the threads in the same block share the **"shared memory"**
- ChatGPT also said **Shared memory** and synchronize

**Shared memory** is an on-chip memory in an SM. And one block runs on one SM. Thus, shared memory holds the data which is shared within one block.
- **SM:** The physical hardware unit inside the GPU that actually executes threads. It's kind of an environment that contains cores, a warp scheduler, and shared memory — and a block always runs entirely on one SM.

Loading and using the data from the memory independently in each thread has a resource-wasting problem caused by duplicated behavior
For example, suppose $C = A @ B$ operation in CUDA. Then each thread would take the operation of each $C[i, j]$. Let's investigate the operation of thread[0,0] and thread[0,1]
- $thread[0,0]$: $C[0,0]$ = $A[0,0] * B[0,0]$ + $A[0,1] * B[1,0]$ + ... + $A[0,n] * B[n,0]$
- $thread[0,1]$: $C[0,0]$ = $A[0,0] * B[0,1]$ + $A[0,1] * B[1,1]$ + ... + $A[0,n] * B[n,1]$

You can notice that both threads load and use the same $A[0,0]$, $A[0,1]$, ... , $A[0,n]$ in their own operation. Loading same $A[0,x]$ data independently is resource waste. So, each thread in the same block stores each data first on the shared memory to save the waste.

How are the values stored in the shared memory? - The thread in $[i,j]$th index stores the data in $[i,j]$th index on the shared memory
- `thread[0,0]` stores data[0,0] in the shared memory
- `thread[0,1]` stores data[0,1] in the shared memory

thread(0,0): tx=0, ty=0 → shared_A$[0][0]$ = A$[0][0]$ store, and output C$[0][0]$  
thread(0,1): tx=1, ty=0 → shared_A$[0][1]$ = A$[0][1]$ store, and output C$[0][1]$  
thread(1,0): tx=0, ty=1 → shared_A$[1][0]$ = A$[1][0]$ store, and output C$[1][0]$

"shared memory store" and "output calculation" both are in a thread.
The thread stores the A in the same index as the thread, and uses shared memory in the output calculation

## 3-1. The advantages of shared memory
Accessing shared memory is faster than HBM
- HBM (GPU main memory): \~ 200 cycles
- Shared memory (on-chip): \~ 5 cycles
- Register (inside thread): \~ 1 cycle
	Shared memory is about 40 times faster than HBM

**We can know that FlashAttention brings us a significant speedup because it moves the calculation place from HBM I/O to the Shared memory**

## 3-2. The reason that ideal data size is `32*n` format
### 3-2-1. There are many kernels for the same operation
1. The kernel with if condition, checking whether the index over the max index
2. The kernel without if condition because the size of data is `32*n`
	- In the case where we choose kernel with if condition, there's overhead of checking if condition for every data(thread).
		- For each dtype → another kernel
		- For each shape → another kernel
		- If the tile is fit → another kernel
		- If the matrix is large → another kernel
			
		Even for the same matmul, there are many kernels and one appropriate kernel is selected

- If the shape is not divisible by 32, then the if contained kernel is selected like 
```
	if (tid < 16):
		a = b + c;  
	else:  
		a = d + e;
```

Warp (32 thread):
- thread 0\~15: execute
- thread 16\~31: waiting until thread 0\~15 execution is done

### 3-2-2. Data alignment
GPU reads memory with a 128-byte chunk at once not 1 byte

When the data is aligned:
- Request: col 0 (memory address 0)
- Read: address 0\~127 at once  ← Reading 128 bytes only one time

When the data is not aligned:
- Request: col 0 (Address 3)
- Read: Address 0\~127, Address 128\~255  ← Need to read memory **Twice**

The start point of data chunk memory address is always multiple of 128
- multiple of 128: 0, 128, 256, ..., 342341120, 342341248, ...

if we read data from address 342341232:
- Chunk that include 342341232: 342341120 \~ 342341247
- Next chunk: 342341248 \~ 342341375

# 4. Grid
It's too deep to study, the purpose of writing this post is to know til the warp and block because when I tried to implement FlashAttention(FA), I was stuck in the two concepts.
If I feel the necessary of grid concept, I'll continue.

# 5. Why the vocab size is padded in `gpt.py`?
It's because of the optimal GEMM(matmul) kernel selection.

If we unpad the vocab_size, then a suboptimal GEMM kernel(overhead with if statement and warp divergence handling code line) would be selected. That kernel contains
(1) An if statement checking whether the thread id is within the boundary
- boundary check if statement
(2) warp divergence would occur (need to handle last tile)
- boundary check masking, reconverge things

But if the vocab_size is divisible by 128, then at first, we don't need to check boundary of thread(data) id. Which means that we don't need if statement at first.

The selected kernel is applied for every data, the kernel is not selected for each of tile(data). So, if we choose inefficient kernel at first, then even the tiles that already have 128 elements and thus don't need to check with if, also have to go through if statement.
- Refer to Section 5.1 in https://arxiv.org/pdf/1909.08053 (https://x.com/ctnzr/status/1623758178587648000)
	- "The original vocabulary size was 50,257, however, to have efficient GEMMs for the logit layer, it is beneficial for the per-GPU vocabulary size to be a multiple of 128. Since we study up to 8-way model parallelism, we pad the vocabulary such that it is divisible by 128 × 8 = 1024, resulting in a padded vocabulary size of 51,200"

## 5-1. How can we check what matmul kernel is used and whether the padded vocab size is fast?

Let's check it with example code.
- The code is run on Google Colab's T4 GPU

### Basic torch profiling

```
import torch
from torch.profiler import profile, ProfilerActivity

A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    C = torch.matmul(A, B)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Output:**
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           aten::matmul         2.76%       7.586ms        97.69%     268.586ms     268.586ms       0.000us         0.00%     785.189ms     785.189ms             1  
                                               aten::mm        63.02%     173.272ms        94.93%     261.000ms     261.000ms       6.543ms       100.00%     785.189ms     785.189ms             1  
                                   cudaFuncSetAttribute         0.40%       1.092ms        19.29%      53.034ms     491.056us       0.000us         0.00%     759.016ms       7.028ms           108  
                                  Lazy Function Loading         1.84%       5.056ms         1.84%       5.056ms      46.819us     706.670ms     10800.00%     706.670ms       6.543ms           108  
                       Runtime Triggered Module Loading        17.55%      48.251ms        17.55%      48.251ms       4.825ms      65.432ms      1000.00%      65.432ms       6.543ms            10  
                                   cudaGetSymbolAddress         9.01%      24.773ms         9.51%      26.139ms      26.139ms       0.000us         0.00%      13.086ms      13.086ms             1  
                                Activity Buffer Request         0.82%       2.254ms         0.82%       2.254ms       2.254ms       6.543ms       100.00%       6.543ms       6.543ms             1  
turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       6.543ms       100.00%       6.543ms       6.543ms             1  
                                  cudaStreamIsCapturing         0.00%      10.275us         0.00%      10.275us       5.137us       0.000us         0.00%       0.000us       0.000us             2  
                                             cudaMalloc         0.23%     624.626us         0.23%     624.626us     124.925us       0.000us         0.00%       0.000us       0.000us             5  
                                               cudaFree         1.50%       4.135ms         1.50%       4.135ms       4.135ms       0.000us         0.00%       0.000us       0.000us             1  
                                 cudaDeviceGetAttribute         0.52%       1.432ms         0.52%       1.432ms      68.177us       0.000us         0.00%       0.000us       0.000us            21  
                                cudaGetDriverEntryPoint         0.00%       1.854us         0.00%       1.854us       0.927us       0.000us         0.00%       0.000us       0.000us             2  
          cudaOccupancyMaxActiveBlocksPerMultiprocessor         0.00%       9.935us         0.00%       9.935us       9.935us       0.000us         0.00%       0.000us       0.000us             1  
                                       cudaLaunchKernel         0.03%      87.136us         0.03%      87.136us      87.136us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize         2.31%       6.343ms         2.31%       6.343ms       3.172ms       0.000us         0.00%       0.000us       0.000us             2  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 274.930ms
Self CUDA time total: 6.543ms
```

### Timing comparison: padded vs unpadded

```
import torch, time

A = torch.randn(8192, 768*2, device="cuda", dtype=torch.float16)
B1 = torch.randn(768*2, 50257*3, device="cuda", dtype=torch.float16)
B2 = torch.randn(768*2, 50304*3, device="cuda", dtype=torch.float16)

def bench(B):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        C = A @ B
    torch.cuda.synchronize()
    return time.time() - t0

print("Unpadded:", bench(B1))
print("Padded:", bench(B2))
```

**Output:**
```
Unpadded: 5.252471685409546
Padded: 3.941844940185547
```

`50257*3` is unpadded (not divisible by 128), `50304*3` is padded (divisible by 128). The padded version is \~25% faster because it selects a more efficient GEMM kernel.

### Full profiling with warmup

```
## WarmUp added
## Torch Profile

import torch
from torch.profiler import profile, ProfilerActivity

A = torch.randn(8192, 768*2, device="cuda", dtype=torch.float16)
B1 = torch.randn(768*2, 50257*3, device="cuda", dtype=torch.float16)
B2 = torch.randn(768*2, 50304*3, device="cuda", dtype=torch.float16)

def bench_and_profile(B, name, warmup=10, iters=20):
    # Warmup: not measured
    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()

    # Profile measured region
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(iters):
            C = A @ B
        torch.cuda.synchronize()

    print(f"\n===== {name} =====")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

bench_and_profile(B1, "Unpadded")
bench_and_profile(B2, "Padded")
```

This is the result of running the above code
===== Unpadded =====

| Name                                             | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| ------------------------------------------------ | --------- | ----------- | ---------- | ------------- | ---------- |
| aten::matmul                                     | 0.000us   | 0.00%       | 6.029s     | 301.462ms     | 20         |
| aten::mm                                         | 5.747s    | 100.00%     | 6.029s     | 301.462ms     | 20         |
| void cutlass::Kernel2<cutlass_75_tensorop_f16... | 5.747s    | 100.00%     | 5.747s     | 287.348ms     | 20         |
| Activity Buffer Request                          | 282.288ms | 4.91%       | 282.288ms  | 282.288ms     | 1          |
| cudaDeviceGetAttribute                           | 0.000us   | 0.00%       | 0.000us    | 0.000us       | 20         |
| cuLaunchKernel                                   | 0.000us   | 0.00%       | 0.000us    | 0.000us       | 20         |
| cudaDeviceSynchronize                            | 0.000us   | 0.00%       | 0.000us    | 0.000us       | 2          |

- Self CPU time total: 5.749s
- Self CUDA time total: 5.747s


===== Padded =====

| Name                                          | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| --------------------------------------------- | --------- | ----------- | ---------- | ------------- | ---------- |
| aten::matmul                                  | 0.000us   | 0.00%       | 5.368s     | 268.417ms     | 20         |
| aten::mm                                      | 5.119s    | 100.00%     | 5.368s     | 268.417ms     | 20         |
| turing_fp16_s1688gemm...                      | 5.119s    | 100.00%     | 5.119s     | 255.945ms     | 20         |
| Activity Buffer Request                       | 249.425ms | 4.87%       | 249.425ms  | 249.425ms     | 1          |
| cudaOccupancyMaxActiveBlocksPerMultiprocessor | 0.000us   | 0.00%       | 0.000us    | 0.000us       | 20         |
| cudaLaunchKernel                              | 0.000us   | 0.00%       | 0.000us    | 0.000us       | 20         |
| cudaDeviceSynchronize                         | 0.000us   | 0.00%       | 0.000us    | 0.000us       | 2          |

- Self CPU time total: 5.121s
- Self CUDA time total: 5.119s

The profiler confirms the kernel name difference:
- **Unpadded**: `void cutlass::Kernel2<cutlass_75_tensorop_f16...` — a less optimal cutlass kernel
- **Padded**: `turing_fp16_s1688gemm...` — the fast Turing tensor-core kernel

We can check that padded one is faster than unpadded one

If the `with` syntax is not familiar, refer to the Context Manager (with syntax) post
