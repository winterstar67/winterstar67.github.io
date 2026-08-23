---
title: "A Tensor Library (ATen)"
date: 2026-07-16
draft: false
math: false
tags: ["PyTorch"]
categories: ["Fundamentals"]
description: ""
---

# 1. ATen
A Tensor library(ATen) is the library of low-level tensor operation implementations that PyTorch uses

## 1-1. ATen flow example 1: ReLU
When PyTorch ReLU is used, the code `y = torch.relu(x)` is executed
But, `y = torch.relu(x)` indicates `aten::relu` which is in the ATen library. 
- `aten::relu` is the name of the operation that PyTorch calls. 
- `aten::relu` means that the ReLU(`relu`) implementation in the ATen(`aten`) library is executed.

## 1-2. ATen flow example 2: PyTorch `nn` operation hierarchy
```
nn.Module
→ torch.nn.functional
→ ATen operation
→ CPU/CUDA kernel
```

Everything up through the ATen operation (including dispatching) runs on the CPU — only the final kernel execution actually runs on the GPU.
- We can check this in Perfetto UI; most GPU processes are named Kernel launch and the other aten operations happen on the CPU.

### 1-2-1. `nn.Linear` execution hierarchy 
```
layer = nn.Linear(128, 64)

layer(x)
└─ F.linear 
	└─ aten::linear
		├─ dispatcher: linear kernel selection - aten::linear+CPU? or aten::linear+GPU? or ...
		└─ Run the selected linear kernel 
			├─ aten::t 
				└─ dispatcher: t kernel selection
			└─ aten::addmm
				└─ dispatcher: addmm kernel selection
					└─ cuBLAS GEMM kernel
						└─ GPU hardware actually run this
```
Even for the same matrix multiplication, there are several implementations considering whether there is a bias, whether the shape is a multiple of 128, and what the dtype is, etc. All of them yield the same results but the calculation efficiency differs among kernels. Dispatcher determines which one to choose based on given information.
- Each implementation is called a kernel. For example,
	- Kernel `A` for matrix multiplication
	- Kernel `B` for the same matrix multiplication
	- ...

Here are some criteria that dispatcher can use
- Whether bias exists
- Whether the tensor dim is 2 or more than that
- Whether the tensor view is contiguous
- Whether the backend is sparse or quantized backend
- Whether the device is CPU, CUDA or something else

Refer to "5. Why the vocab size is padded in `gpt.py`?" in {{< wikilink "cuda" >}}

### 1-2-2. `nn.Linear` vs `nn.functional.linear` vs `aten::linear`
Two modules have the same operation which is matrix multiplication. But there is a difference in terms of state.
- `nn.Linear`: It saves the weight and bias. `nn.Linear` is an object, not just a function
- `nn.functional.linear`: It doesn't save weight and bias but just does the calculation.
- `aten::linear`: This is a `C++`-level operation that `nn.functional.linear` calls. This is the point where dispatcher's kernel selection happens.

## 1-3. Some ATen operations list

| ATen operations | Meaning                            |
| --------------- | ---------------------------------- |
| `aten::add`     | Element-wise add                   |
| `aten::mul`     | Element-wise multiplication        |
| `aten::matmul`  | Matrix multiplication              |
| `aten::mm`      | 2D Matrix multiplication           |
| `aten::bmm`     | Batch Matrix multiplication        |
| `aten::addmm`   | Matrix multiplication and Bias add |
| `aten::linear`  | Linear layer operation             |
| `aten::conv2d`  | 2D Convolution                     |
| `aten::relu`    | ReLU                               |
| `aten::softmax` | Softmax                            |
| `aten::reshape` | Shape change                       |
| `aten::copy_`   | Tensor copy                        |
### 1-3-1. Naming convention
```
aten::add # Stores the result in a new tensor allocating new memory.
aten::add_ # Stores the result in the existing tensor
```
