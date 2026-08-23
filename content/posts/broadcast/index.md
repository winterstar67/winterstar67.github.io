---
title: "Broadcast"
date: 2026-03-28
draft: false
math: false
tags: ["Pytorch", "Operation"]
categories: ["Fundamentals"]
description: ""
---

# Terminologies
I'll describe:
1. length of shape as `rank` or `ndim`
2. Each axis as `dim`
3. the number of i-th dim element as `size`

For example, if `tensor.shape = (1,2,3,10)`, its ndim is 4 (there are four dims) and fourth dim size is 10.

# 1. Function of broadcast
Broadcast is a function that fits the two operands in elementwise operation.
This function is widely and frequently used when we work with pytorch or numpy.
But, if you don't know the rule of broadcast mechanism, it could be tricky to debug the codes.
You should understand what's replicated and how it acts with which rules.

The rule of applying broadcast is super simple.

# 2. Broadcast mechanism
I'll describe the process of broadcast with the below example to be followed easily

## 2-1. Right re-arranging the operand with smaller rank
If the ranks of tensor arrays are not same, the smaller one must (1) expand the rank to match the larger one, (2) be right re-arranged and (3) add 1s in each added dim at left side

Let's check a process with the example, `[B, T, H, D]` + `[T, H, D]`,
1. Expand the rank of `[T, H, D]`, which is smaller, from 3 to 4
2. Push `[T, H, D]` into right side so that it becomes `[1, T, H, D]`
	- It's right arranged and 1 is at the newly added dim which is at left

## 2-2. Compare each dim and duplicate the tensors
When the two operands have same `ndim` but the sizes of two tensors in i-th dim are different, one of the sizes should be 1.

Here's the example, `[B,T,H,D]` and `[B,1,H,D]`:
- `1` in `[B,1,H,D]` becomes `T` by repeating `[H,D]` matrix (the right dim of 1 part) and it becomes `[B,T,H,D]`

Not allowed pattern: `[B,2,H,D]` and `[B,4,H,D]`
- Even if one of the dims is divisible by the other, it's impossible to broadcast.

# 3. Exercise
Answer to the below case
#### Answer the shape of the result of element-wise multiplication of two matrices
```
A shape: (5, 1, 4)
B shape: (3, 4)
```

# 4. Things to Know
The broadcast uses `.expand()` which means it doesn't copy data and doesn't allocate a new memory address.
