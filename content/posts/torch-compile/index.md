---
title: "torch.compile"
date: 2026-06-14
draft: false
math: false
tags: ["Efficiency"]
categories: ["Fundamentals"]
description: ""
---

# 0. Background
- Eager mode: It executes the operation right after it encounters the code. To run the next code, Program Counter(PC) must move to the next code line
	- Eager mode is the basic mode of PyTorch
	- `torch.compile` doesn't act in eager mode. It prepares the graph of operations at the first run and then runs it in the faster way than eager mode.

# 1. `torch.compile`
`torch.compile` is a module to run PyTorch code faster than eager mode
- Eager mode executes operations one by one in order

# 2. Effects of using `torch.compile` compared to not using it
Before encountering the next code, Python doesn't know what would come next which could be add, matmul, or relu.
But `torch.compile` knows the order of operations, and can prepare more efficiently grouped kernels 
## 2-1. The number of read/write to the HBM is decreased
- For the following Python code
- ```
	a = x + 3
	b = a * a
	z = relu(b)
  ```
  Without `torch.compile`:
- ```
	read x from HBM  
	write a to HBM  
	  
	read a from HBM  
	write b to HBM  
	  
	read b from HBM  
	write z to HBM
  ```
  With `torch.compile`:
  ```
	read x from HBM  
	compute x + 3  
	compute square  
	compute relu  
	write z to HBM
  ```

## 2-2. Kernel integration
- ```
  Kernel 1: matmul, writes intermediate tensor Z  
  Kernel 2: add b, reads Z, writes intermediate tensor A  
  Kernel 3: relu, reads A, writes output Y
  ```
  Becomes
- ```
  Kernel 1: matmul
  Kernel 2: fused add + relu
  ```
  . The number of kernels is decreased

## 2-3. The number of calling Python is decreased
- For the `y = torch.relu(x @ w + b)` code,
- Without `torch.compile`:
	- ```
	  Python → matmul kernel call
	  Back to Python
	  Python → add kernel call
	  Back to Python
	  Python → relu kernel call
	  Back to Python
	  ```
- With `torch.compile`:
	- ```
	  Calling one kernel that handles one large graph of relu(matmul(add))
	  Back to Python
	  ```

## 2-4. Speed
First run with `torch.compile` would be slower than not using it. But after that, if we use the cached compiled representation, it would be faster than not using it.

# 3. Re-compile and cache hit conditions
## 3-1. Torch tensors
- If only the shape, dtype, device of tensors are the same as the cached one, then the cache is hit (There's no recompile)
- In the case that keeps the shape, dtype, device of tensors but the internal value of a tensor is changed, still it's not recompiled
## 3-2. Python Integer
- If the internal value in a Python integer object is changed, then it would be recompiled.
	- The `red_dim` in `optim.py` is the case of a Python integer. `red_dim` is either -1 or -2. So recompiling occurs at most one (-1 after -2 case or -2 after -1 case) and `torch.compile` reuses the cached -1 and -2 cases.

# 4. Difference of `torch.compile` and C++ compiler
## 4-1. C++ compiler (AOT compilation)
C++ compiler uses AOT compilation. It optimizes the code before running it.
At run time, the code is executed faster than an interpreter would because of the pre-defined variable types.
It creates an additional file which is machine readable and runnable by translating the C++ code into machine language code.

## 4-2. `torch.compile` process (JIT compilation)
`torch.compile` uses a JIT compiler. It optimizes the code during the execution of code.
- If the code is already optimized (compiled) by JIT, then it's cached in memory. So there would be no compilation.
- If the code is not cached, during the execution, there would be a process of compilation
- TorchDynamo checks whether the executed code is cached or not.
During run time, `torch.compile` creates the compiled executable representation.
It contains the following steps

1. TorchDynamo (Capturing the graph groups) using FX graph representation (How to express the graph)
	- There's code that breaks the graph. For example, if there's code like `print`, it's not a target of compiling (grouping as a graph). So the code before and after `print` would be disconnected. So there are several groups of graphs - a large graph and a small graph.
	- The guard, which is the implementation in TorchDynamo, detects whether the code is cached or not

2. TorchInductor (Design how to optimize the graph)
	- The graph detected by TorchDynamo is handed over to TorchInductor.
	- TorchInductor optimizes the graphs
	- Usually, the pointwise kernels are integrated.

# 5. Graphs in PyTorch
Autograd Graph and Graph by `torch.compile`(FX Graph) are different
- What Autograd Graph stores: The middle values of activations during forward pass for backpropagation to compute gradients
- What FX Graph stores: The compiled(kernel fusioned, etc.) graph
