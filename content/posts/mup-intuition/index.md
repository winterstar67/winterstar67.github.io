---
title: "(Errors found, revision needed) muP intuition"
date: 2026-07-04
draft: true
math: true
tags: []
categories: ["Fundamentals"]
description: ""
---

**This post currently contains known errors and will be corrected.**

# Papers
**I didn't read whole of these papers. Reading all pages too long and it's quite hard to understand**
## Feature Learning in Infinite-Width Neural Networks
- **Title**: *Feature Learning in Infinite-Width Neural Networks*
- **Authors**: Greg Yang, Edward J. Hu
- **URL**: https://arxiv.org/pdf/2011.14522
- **Length**: 65p
- This post is mainly described about the µP definition
- This paper proposed µP parameterization explaining what problems exist on traditional SP method and how the µP solves the problem of SP
	- But the pages of paper and mathemetical lemma and theorm are too many, so I couldn't read and understand the whole contents. 
		- I only read Section 4(SP) and Section 5(µP) just to catch the intuition

## Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer
- **Title**: *Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*
- **Authors**: Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao
- **URL**: https://arxiv.org/pdf/2203.03466
- **Length**: 48p
- This post hardly describe about the content of this paper
- This paper shows that µP parameterization can transfer the LR of small model into the larger one
	- This makes the cost of searching LR value in large model by converting the problem of large model LR search into small model's search

# Background
- Standard Parameterization(SP)

# 1. Motivation
## 1-1. Standard Parameterization
I didn't fully understand the whole Lemma and proves, but one MLP examples gave me an intuition why the traditional SP has a limitation.
- **The key limitation is that there is dilemma between training loss and feature learning.**

Suppose there's 2 layered MLP with following settings
- Weight distributions of input layer \~ $N(0, 1)$
- Weight distributions of hidden layers \~ $N(0, \frac{1}{n})$
- Weight distributions of output layer \~ $N(0, \frac{1}{n})$
- Optimizer: SGD
- $H = W@X$ 
	- $H$: activation
	- $W$: Weight matrix
	- $X$: Input or activation
- $f = V@H$
	- $f$: loss function
	- $V$: Weight matrix in last layer

SP parameterization is motivated from preventing to blow up activation values by increasing width `n`.
Suppose the linear combination operation is `y=W @ X`.
Then, each feature activation is derived from the dot product of column vector of $X$ and row vector of $W$. Each vector has width `n`. So,
- The activation value in next layer is $\sum_i^n{x_i w_i}$.
- Mean of activation is 
$$E[x_1w_1 + x_2w_2 + ... + x_nw_n] = $$
$$E[x_1w_1] + E[x_2w_2] + ... + E[x_nw_n] = E[x_1]E[w_1] + E[x_2]E[w_2] + ... + E[x_n]E[w_n] = $$
$$0*E[x_1] + 0*E[x_2] + ... + 0*E[x_n] = 0$$
- The variance of activation
$$Var[x_1w_1 + x_2w_2 + ... + x_nw_n] = $$
$$Var[x_1w_1] + Var[x_2w_2] + ... + Var[x_nw_n] = Var[x_1]Var[w_1] + Var[x_2]Var[w_2] + ... + Var[x_n]Var[w_n] = $$

Because $Var[wx] = E[w^2]E[x^2] - E[w]^2E[x]^2 = E[w^2]E[x^2] - 0*e[x]^2= Var[w]E[x^2] = \frac{1}{n}*E[x^2]$,

$$n * \frac{1}{n} * E[x^2]$$
How large the width `n` of layer is, keeping the variance of activation makes the forward stable.

## 1-2. The problem of SP
As described, the forward right after initialization is not a problem.
The problem of SP is after update.
```
- Initial state:      Okay
- Forward in step 0:  Okay
- Backward in step 0: Okay
- Forward in step 1:  NOT Okay
```
### 1-2-1. Backward in step 0
$W_{step=1} = W_{init} - LR*G*X^T$
- $G$: Global gradient from loss function. In MLP, it's chain of weights matrix multiplication
- $X^T$: Local grad. $\frac{dW@X}{dW} = X^T$

### 1-2-2. Forward in step 1
#### 1-2-2-1. Feature learning - Activations
$$H_1 = W_{step_1}@X = (W_{init} - LR*G@X^T)X = W_{init}@X - LR*G@X^T@X = H_{init} - LR*G@X^T@X$$
$$H_1 - H_{init} = - LR*G@X^T@X$$
The scale of $H_1 - H_{init}$ is important. To get the scale, each component's scale need to be calculated.
- $H_1 - H_{init} = LR*G@X^T@X$: It has an variance $\theta(n^{-c + \frac{1}{2}})$. If we increase the width $n \rightarrow \infty$, it would be 0 if c<0, which means there's no feature learning.
	- LR: It's a variable. Suppose it's $n^{-c}$
	- $G$: Global gradient is the continuously multiplied weight matrices. G \~ $N(0, \frac{1}{n})$
		- Suppose the global gradient is compose of four weight matrix `W_1 @ W_2 @ W_3 @ W_4`. Every weight matrix is initialized with $N(0,\frac{1}{n})$.
		- `W_1 @ W_2`: Because both matrices are from $N(0, \frac{1}{n})$, so the dot product (each element in the result matrix) would be form of $w_{11}w_{21} + w_{12}w_{22} + ... + w_{1n}w_{2n}$  when the $w_i$ is the element of $W_i$. $$Var(w_1w_2) = E[w_1^2w_2^2] - E[w_1w_2]^2 = E[w_1^2]E[w_2^2] - (E[w_1]E[w_2])^2 = $$$$Var(w_1)Var(w_2)-0 = \frac{1}{n}\frac{1}{n} = \frac{1}{n^2}$$
		- There's `n` summation of $w_1w_2$. So, $n*Var(w_1)*Var(w_2) = \frac{1}{n}$. Therefore, $W_1@W_2$ \~ $N(0,\frac{1}{n})$.
		- We can recursively apply this on $W_3$ and $W_4$ too. 
		- So, **G \~ $N(0, n^{-1})$**.
		- The scale of G is RMS = $\sqrt{E[g^2] = Var[g] + E[g]^2 = Var[g] + 0 = n^{-\frac{1}{2}}}$
	- $X^T@X$: If `X` is input, the dimension that's dot producted would be supposed to be fixed with `d`. If `X` is activations, the shape would be `n`.
		- `X` = Input matrix case: $X^T@X$ element is $x^2$. Each $x^2$'s variance is irrelevant to width `n`, so the variance is $n * \theta(1) = \theta(n)$.
		- `X` = activation matrix  case: $W@X$ has variance value $n*\theta(\frac{1}{n})*\theta(1)=\theta(1)$. Because the activation has $\theta(1)$ variance, $n * \theta(1) * \theta(1) = \theta(n)$

#### 1-2-2-2. Loss decrease
The loss is expressed as $f = V*H$.
Authors set the other parameters including $V$ is not updated except the updated $W_1 = W_{init}$.
$f_1$ is loss after step 1 update.
$$f_1 = V_1@H = V_1@(W_1 @ X) = (V_{init} - n^{-c}*G@H) @ (W_{init}-n^{-c}*G@X^T)@X$$
If $c=1$, then $V_1$ has same scale with $V_{init}$ because the $n^{-c}*G@H$ term has $\theta(1)$ scale. So for the simplicity, I'll replace $V_1$ into $V_{init}$.
$$f_1 = V_{init}@H = V_{init} @ (W_{init}-n^{-c}*G@X^T)@X = V_{init}@W_{init}@X - V_{init}@(-n^{-c}*G@X^T)@X$$
It's $f_1 = f - V_{init}@(-n^{-c}*G@X^T)@X$ $\rightarrow$ $f_1 - f = n^{-c}*V_{init}@G@X^T)@X$.
- $f_1 - f$ means the difference of loss value after one update.
- $X^T@X$ element has $\theta(B)$ magnitude with size $n$, B is batch size.
	- If $xx$ is one element in $X^T@X$, it has $\theta(B)$ scale and the row and column size is $n$ each.
- $G$ has $\theta(\frac{1}{n})$ magnitdue.
- $V_{init}$ has $\theta(\frac{1}{n})$ magnitude.
- $V_{init}@(-n^{-c}*G@X^T)@X$ has $n^{1-c}$

When width $n \rightarrow \infty$, 
- If c>1, loss after weight update wouldn't be changed.
- If c<1, loss after weight update would blow up.
- If c=1, loss after weight update would not blow up being changed

So, appropriate c value is 1. **But as we saw in feature learning case, c=1 causes no feature learning.**
##### It's the dilemma of feature learning and loss decrease in SP method.
When $n \rightarrow \infty$,
- If $c=1$, there's no feature learning.
- If $c=\frac{1}{2}$, loss would be blown up

### 1-2-3. Intuition over the simple 2-layer MLP 
#### 1-2-3-1. For the several steps
The format of equation is `W_t = W_init − LR⋅g0 − LR⋅g1 − ⋯ − LR⋅gt−1.
There's no difference of Weight scale because it's addition, not multiplication
- But after several steps, probably we should care the dependency between activation and weights because the activation values after 1 step is calculated based on weight that's updated based on other layers too.

#### 1-2-3-2. For the several depth
I already wrote this.
G is global gradient and it is consist of multiplication of weight matrices in several layers.


# 2. Proposed parameterization method: µP
## 2-1. The key different settings of µP and SP
1. The variance of last layer's weight distribution is $\frac{1}{\sqrt{n}}$
2. `c` = 0 in µP, which means $n^{-c = 0}$ which means the LR is irrelevant to model width `n`

By multiplying $n^{-\frac{1}{2}}$ on the last layer, there are two changes
1. The non-last layers have $n^{-\frac{1}{2}}$ multipled scale on their forward activation. $n^{-\frac{1}{2}}$ would be reflected on the update amount
	- This makes $\theta(n^{\frac{1}{2}-c})$ to be $\theta(n^{-c})$
2. Last layer have $n^{-1}$ multipled scale on their forward activation. $n^{-\frac{1}{2}}$ would be reflected on the backward update amount, and the other $n^{-\frac{1}{2}}$ is from the change that we multiply $\frac{1}{\sqrt{n}}$ on µP
	- This makes $\theta(n^{1-c})$ to be $\theta(n^{-c})$

## 2-2. How the µP solves the dilemma of SP
Suppose the loss is $f = \frac{1}{\sqrt{n}} V@H$ instead of $f = V@H$. There's $n^{-\frac{1}{2}}$ multiplication comparing to the SP method.
One step of update would be $W_1 = W_{init} - LR*n^{-\frac{1}{2}}*G@X^T$.

### 2-2-1. Expression of feature learning and loss decrease after update
#### 2-2-1-1. Feature learning
The equation of activation value is the followings
$$H_1 = W_1@X = (W_{init} - LR*n^{-\frac{1}{2}}*G@X^T)@X = H_{init} - n^{-c}*n^{-\frac{1}{2}}*G@X^T@X$$
- $H$ has $\theta(n)$ scale,
- $G$ has $\theta(n^{-1/2})$ scale - RMS is the scale.
- $X^T@X$ has $\theta(n)$ scale

The scale of $H_1 - H_{init} = n^{-c}*n^{-\frac{1}{2}}*G@X^T@X$ is $n^{-c}$

#### 2-2-1-2. Loss decrease
The equation of activation value is the followings
$$f_1 = \frac{1}{\sqrt{n}}V@H_1 = \frac{1}{\sqrt{n}}V@(H_{init} - n^{-c}*n^{-\frac{1}{2}}*G@X^T@X) = f - \frac{1}{\sqrt{n}}*n^{-c}*n^{-\frac{1}{2}}*V@G@X^T@X$$
- $H$ has $\theta(n)$ scale,
- $V@G$ has $E[vv^T] = n*E[v^2] = n*Var(v) = n*\frac{1}{n} = \theta(1)$ and $Var(vv^T) = Var(v^2) = n*2*(\frac{1}{n})^2 = \theta(n^{-1})$ scale each - RMS is the scale.
- $X^T@X$ has $\theta(n)$ scale

The scale of $n^{-c}*n^{-\frac{1}{2}}*G@X^T@X$ is $n^{-c}$

When we set $c=0$, both feature learning and loss decrease dilemma can be solved.
Also, $c=0$ means that even if the width of model is large, we can use same LR that's used in smaller model when we use µP parameterization.
- We can set LR value regardless the model width

##### Things to keep in mind
- Scale is RMS = $\sqrt{E[x^2] = Var(x) + E[x]^2}$.
	- But in the most case in this post, the $E[x] = 0$. So, std is considered as RMS.
- Independency between $V@G$ and $X^TX$.
	- $Var(V@G@X^T@X)$ can be split into $Var(V@G)$ and $Var(X^T@X)$. So the variance or scale estimation was convenient.
	- But this dependency is valid before step 2 update.

# The properties not covered in this intuition purpose post
- After update 1 step case.
	- When the update step is more than 1, The dependency of $V@G$ and $X^T@X$ would be broken
- For the various optimizer such as AdamW or Muon, etc.
- The other layers except MLP such as Conv layer or non-linear activation function
- When the V is also updated. (The assumption in this post it that only the W is updated)
