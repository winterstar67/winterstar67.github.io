---
title: "Execution Time Debug with Pytorch"
date: 2026-07-26
draft: false
math: true
tags: ["PyTorch", "Estimation", "Analysis"]
categories: ["Fundamentals"]
description: ""
---

# 1. Execution time debug
## 1-1. Things to run at first
### 1-1-1. Warm-up
The first few executions could be slower than the usual ones because of the following initialization steps:
- CUDA initialization
- GPU kernel loading (Caching)
- Memory allocation
- cuDNN algorithm selection
- `torch.compile` graph compile
If we estimate the latency without a warm-up, the latency result could be longer than the usual one
```
# GPU warming up
for _ in range(warmup_steps): 
	_ = model(inputs)
```

### 1-1-2. `torch.cuda.synchronize()`
This function waits until all GPU work is done, then runs the next code.
Not to be affected by the asynchronous property of the GPU, I start to run `torch.cuda.synchronize()` before running the `time.perf_counter()` function.

## 1-2. Estimation Tools
### 1-2-1. `time.perf_counter()` - End-to-end time estimation
It measures how much time has elapsed including both the CPU time and the GPU time.
But, GPU is executed asynchronously in the background of the Python process, which means it could run the next code line before the GPU process is done
```
# Example of simple timer problem
start = time.perf_counter()
output = model(inputs) # Start the asynchronous GPU process and move to the next line
end = time.perf_counter() # (Probably) Run `perf_counter()` before the GPU process is done
	# It's not certain whether the `perf_counter()` is run before the GPU process is done because calling `perf_counter()` is independent of the GPU process.

# `end - start` The result would not contain GPU process time
```

To include GPU process time in the elapsed end-to-end time, we have to force the `end = time.perf_counter()` to ensure it runs after the GPU process is done.
`torch.cuda.synchronize()` function makes the code execution wait until the GPU process above is done.
The example code below measures latency including GPU process time.
```
start = time.perf_counter()
output = model(inputs) # Start the asynchronous GPU process and move to the next line
torch.cuda.synchronize()
end = time.perf_counter() # Run `perf_counter()` after the GPU process is done

latency_ms = (end-start)*1000
```

### 1-2-2. `torch.cuda.Event(enable_timing=True)` - Only GPU time estimation 
`starter = torch.cuda.Event(enable_timing=True)` creates an empty object.
When we call `starter.record()`, now it's not empty anymore. It records the event (The event means a counter; it measures how much the counter's value has increased from a certain reference point).
We should place the `.record()` execution code right before/after the GPU execution code like the code below.  
```
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

starter.record()
_ = model(inputs)
ender.record()

latency_ms = starter.elapsed_time(ender)
```
This code would make the format of the GPU execution queue like `[starter event record, GPU task 1, GPU task 2, GPU task 3, ..., ender event record]`.
When we need to estimate this repeatedly for several data or steps, we don't need to use `torch.cuda.synchronize()` because the `record()` is also run on the GPU like the other GPU processes.
- The time recorder, `ender event record` in `[starter event record, GPU task 1, GPU task 2, GPU task 3, ..., ender event record]`, runs only after all the preceding GPU tasks in the queue have finished. So, when we estimate just GPU execution time using `torch.cuda.Event(enable_timing=True)`, we don't have to use `torch.cuda.synchronize()` unlike the `time.perf_counter()`.

`time.perf_counter()` includes every event during the inference including non-GPU process time, not just GPU process time.
- `time.perf_counter()` is a better fit for real-world environments

# 2. Throughput
Throughput is another metric that measures how much data a model can process per unit of time.
When the data size differs, then the latency would differ too.
So to make the latency comparable, $\text{Throughput} = \frac{\text{Batch size}}{\text{Latency in seconds}}$ is used.
Throughput means how much data can be processed per second

# 3. Statistics
For a more precise estimation, it would be better to consider the following statistics of the time measurement
- **Mean**
- **Std**
- **Median or p50**
- **p95**
- **p99**
- **Min**
- **Max**

# 4. Situations to analyze
- The training time.
- Inference time and validation time.
- A part that takes the most time. 
- The longest part in each situation.
