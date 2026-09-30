---
title: "Distributed training (Multi GPU)"
date: 2026-06-12
draft: false
math: true
tags: ["PyTorch", "Distributed training"]
categories: ["Fundamentals"]
description: ""
---

# 0. Background
- Every GPU runs exactly the same code, processes run by every GPU are independent of each other.
- If we run `train.py` code file with 8 GPUs, then each GPU runs exactly the same code file `train.py`.
- The one thing that's different between GPUs is the RANK.
- The way for GPUs to communicate with each other is `torch.dist`.
{{< wikilink "distributed-training-environments" >}}
- Concept of `RANK` and how to use it.

# 1. GPU comparison
## 1-1. One GPU(single process)
One GPU(one worker) runs one code file(one process).
This is the way that we run the code usually.
We run the code file only one time. There is only one process.
The model parameters, forward activations, and backward gradients each exist only once. There's no duplicated data.

## 1-2. Multi GPU(several process)
This is not about splitting which GPU would run first part of code or last part of code.
**Every GPU runs exactly the same code file in its own process**.
If there are four GPUs and the GPUs run the code file below, then here's what happens

# 2. Running Example
## 2-1. Settings
### 2-1-1. GPUs
- GPU0
- GPU1
- GPU2
- GPU3

### 2-1-2. Code
```
# test.py
print(1)
print("a")
```

### 2-1-3. Run
`torchrun --nproc_per_node=4 test.py`

### 2-1-4. Result
```
# Terminal Output example
# The print order can be changed
1
1
1
1
a
a
a
a
```

There are four processes of `test.py`.
```
# test.py
print(1)
print("a")
```
Each GPU is allocated one process without duplication
- Think each GPU takes its own process without duplication

# 3. How to run multiple GPUs?
`torchrun --nproc_per_node=NUM_GPU py_file`
- `--nproc_per_node`: number of processes per node
This command creates NUM_GPU `py_file` processes.
Not only that, this command allocates one GPU per process.
So, basically all of them have the same data such as the same model parameter **Because they run the same code**

# 4. If they run exactly the same code, then does each GPU generate exactly the same results, so there is no difference compared to a single GPU?
If we don't use a method for GPUs to communicate across the others, then NO. There is no difference at all.
But if we can control the results from each GPU across all GPUs, then running the code with multi-GPU has a meaning.
One of the common techniques is distributing the data into each GPU keeping them to have same model parameters.
Then there could be a question of "Which GPU's model parameter is selected to save the model parameter?"
To remove this concern, we need to synchronize the gradients in each GPU to be same so that the update value and updated model parameters to be same.
- This could be done by `all_reduce` things averaging all gradients across the GPUs

# 5. How each GPU handles different data
Even when they run the same code, there's one value that's different in each GPU(process), which is **`RANK`** in os environment.
Suppose the data is loaded on `data` variable. Even in the same code across each GPU process, `RANK==0` in GPU 0, `RANK==1` in GPU 1, etc.
- Refer to {{< wikilink "distributed-training-environments" >}}
Then like `shard_data = data[RANK*chunk_size : (RANK+1)*chunk_size]`, now the data across the GPU is different by each GPU.
Each GPU runs the code and obtains its own forward activations and backward gradients from its own data.

## 5-1. How it utilizes every gradient on the model update
Splitting the data, is that all? No, we need to utilize it. If we just let them end in independent process without integrating it or mixing them in somehow, then it's just the independent process like four different model training.
- Different data can be selected using `RANK`

We need synchronization and utilization because the model that we train is only one model.
If we average the gradients across every GPU and update the model with averaged gradients, then it has an effect of getting the gradient of bunch of data that is not affordable with a single GPU due to memory limitations.
The `torch.dist` makes it possible for GPUs to communicate with each other like averaging gradients of each GPU.

Compared to a single GPU, the n-GPU case has the following advantage.
1. When the maximum batch size at once is `batch_size` in a single GPU, we can process a batch size of `batch_size * n`
2. By the effect that we can handle bunch of data samples at once, one epoch iteration could be done faster than a single GPU case

The library we use for communication of GPU is `torch.dist`, but the library that processes actual background communication is NVIDIA Collective Communications Library **(NCCL)**
- `torch.dist` is just API to call NCCL

# 6. Flow of distributed training
There are several ways to implement distributed training
One of the popular ways is DDP, which is the concept explained above

Suppose each GPU loads exactly the same weights, but uses different data. Here's a simple example
## 6-1. Initial settings (Loading Data and Model parameter)
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

## 6-2. Forward and backward
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

## 6-3. Average gradients (Synchronization, Communication across the GPUs)
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

## 6-4. Weight update
- GPU1: 
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W'_{2}, W'_{3}]$
- GPU2: 
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W'_{2}, W'_{3}]$
- GPU3:
	- Original Weights: $[W_1, W_2, W_3]$
	- Updated Weights: $[W'_{1}, W'_{2}, W'_{3}]$
