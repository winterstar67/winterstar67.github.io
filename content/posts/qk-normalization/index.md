---
title: "QK Normalization"
date: 2026-04-09
draft: false
math: true
tags: ["Paper", "stability"]
categories: ["Fundamentals"]
description: ""
---

# Paper info
- **Title**: *Query-Key Normalization for Transformers*
- **Authors**: Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, Yuxuan Chen
- **URL**: https://arxiv.org/pdf/2010.04245  
- **Length**: 8 pages

# 0. Background
## 0-1. Dot product
- ${Q}\cdot{K} = |Q| |K| \cos{\theta}$
- The magnitude of the dot product is determined by the l2 norm of Q and K and the $\cos\theta$ 
## 0-2. Softmax function
- Suppose the loss is $L$
1. Forward pass: $y_i = \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}}$ ; z: logits (the input of softmax) 
2. Backward pass: The gradient from softmax($y$) output to element($z_i$) is as follows
	- $\frac{dL}{dz_i}$ = $\sum_j{\frac{dL}{dy_j}*\frac{dy_j}{dz_i}}$
	- If $j=i$, then $\frac{dy_{j=i}}{dz_i} = y_i(1-y_i)$ 
	- If $j\not=i$, then $\frac{dy_j}{dz_i} = -y_i(y_j)$
- $\sigma(z) = \sigma(z+c)$ (c is a constant)
- `F.rms_norm(x, (x.size(-1),))`: normalizing tensors aligned to the second argument tuple dimension direction
- The second argument of `F.rms_norm` means "Which dimension would you apply the normalization?"
	- The normalized_shape argument always checks starting from the last dimension of input x (If the shape is (B,T,H,D) then, check from D to B order)
	- So, `F.rms_norm(x, (x.size(-1),))` is always natural because the rms_norm applies from the last dimension and x.size(-1) indicates the last dim.
	- If you'd like to apply it for the last two dims, then you should write it in `F.rms_norm(x, (x.size(-2), x.size(-1)) )`
	- Side note (unrelated to rms_norm): The (last_dim,) style argument can't handle the batch_normalization because the batch norm is operated only on the first(batch) dimension direction.
		- If we apply like `F.rms_norm(x, (x.size(-2), x.size(-1)) )`, it doesn't mean we apply to the last dim direction and then the second-to-last dim direction separately, we apply the normalization treating the last two dims as a single unit of normalization(as a single normalization block range).
			- In a matrix with (m,n), if we give (last_dim,), then it applies the normalization per each row - n elements
			- But if we give (second_to_last_dim, last_dim), then it applies the normalization for the whole $m * n$ values in the matrix once. - all $m * n$ values in the matrix are used for l2 norm(squared, summed, mean, sqrt) all together
		- But the batch norm must apply the normalization only on the first dimension, not like rms_norm which applies from the last dimension
- The syntax of creating a tuple is `(D,)`, `(H,D)` format. The comma should be right most. `(,D)` is wrong syntax

# 1. Motivation
## 1-1. Preventing the saturation of softmax.
- Why is the saturation of softmax bad?
- Saturation means that only one element is close to 1 and the others are 0 like a one-hot vector.
- In saturation state, every term $\frac{dL}{dy_j}*\frac{dy_j}{dz_i}$ in $\sum_j{\frac{dL}{dy_j}*\frac{dy_j}{dz_i}}$ (Gradient) becomes close to 0,
- For the $y_i=1$
	- If $j=i$, then $\frac{dy_{j=i}}{dz_i} = y_i(1-y_i) = 1(1-1) = 0$
	- If $j\not=i$, then $\frac{dy_j}{dz_i} = -y_i(y_j) = -1*0 = 0$
- For the $y_i=0$
	- If $j=i$, then $\frac{dy_{j=i}}{dz_i} = y_i(1-y_i) = 0(1-0) = 0$
	- If $j\not=i$, then $\frac{dy_j}{dz_i} = -y_i(y_j) = -0*1 = 0$
- This means the gradient doesn't flow well.
- So preventing the saturation of softmax is important for stable training. 
- This is what the authors were trying to solve.
## 1-2. It forces the Model to train the relationship between q and k based on angle except the l2 norm of q and k.
- The fundamental component that affects the output of attention is ||q||, ||k||, and cos(theta)
- One of the motivations of adapting attention methodology is giving high attention to tokens that are highly related to each other for predicting the next token.
- Just giving a big attention value for the token which has a large l2 norm can happen in classical attention mechanism.
	- That's irrelevant to the relationship between two tokens. Having a large absolute value of l2 norm would always have the largest attention which ignores the relationship and has no representational flexibility. The model needs to predict the output based on the relationship of input tokens
	- For every query token, the key token which has a large l2 norm could have a high attention value (Static)

# 2. Explanation of this paper
The purpose of this paper is the following:
    **2-1**. Preventing the saturation of softmax
    **2-2**. Converting the attention value to be dependent only on the cos similarity, excepting the norm of q and k

## 2-1. Preventing the saturation of softmax part
### 2-1-1. What would happen if the saturation of softmax occurs in the forward pass and the backward pass?
This is explained in the Motivation section above.

### 2-1-2. We care about this phenomenon because the classical Transformer model suffers this problem.
In transformers, the scale of $||p||$ and $||k||$ is unlimited, so the problem of attention saturation was worse.
- Still, even if the difference between the elements in the array is small, after the softmax, the saturation can happen. Can we say that the qk norm can solve the saturation?
- The range of difference between two elements has a maximum limit of $[-1, 1]$ when we apply QK norm.

### 2-1-3. Then, how does the QK norm can relieve this problem?
The important phenomenon is that now there's a limit on the difference between any two logits pair from two tokens which is $[-1, 1]$ 
- So the maximum difference is 2 from (1 - (-1))

## 2-2. Converting the attention value to be dependent only on the cos similarity, excepting the norm of q and k
In terms of interpretation, only the cos term is left, which makes the situation much more controllable
Just simply think. "Then is it reasonable to multiply a fixed, identical scalar or just to multiply 1 for every layer? A learnable scalar can still cover these cases too — even if a static multiplier turns out to be optimal, training could converge to that same fixed value."

# 3. QK norm methodology
## 3-1. How to calculate
Simple. Just dividing ||q|| and ||k|| by a learnable scalar can give the above effects
- learnable scalar: we need to control the scale of $\cos\theta$

## 3-2. How to Implement
```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

cos, sin = cos_sin

q = F.rms_norm(apply_rotary_emb(q, cos, sin), (q.size(-1),))
k = F.rms_norm(apply_rotary_emb(k, cos, sin), (k.size(-1),))
attention = F.softmax(self.scalar_qk * torch.matmul(q, k.permute(0,1,3,2)))
```

But in nanochat, the learnable scalar isn't used. Just fixed 1.2 was multiplied instead of a learnable scalar
- According to the comment in the code, it's written that $*1.2$ is for making softmax sharp.
- The exact reason for choosing to sharpen the softmax isn't stated, but it was likely determined empirically.

The softmax becomes sharper when the difference between two elements becomes larger.
- For the `[0.5, 0.2, 0.3]`, if we multiply `2`, it would be `[0.5 + 0.5, 0.2 + 0.2, 0.3 + 0.3]` `->` The difference becomes larger, so the softmax becomes sharper. That's it.
- Or, with difference view, `0.5-0.2 = 0.3` becomes `(0.5-0.2)*2 = 0.6`
