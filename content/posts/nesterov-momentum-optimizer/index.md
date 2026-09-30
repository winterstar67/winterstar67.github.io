---
title: "(Errors found, revision needed) Nesterov Momentum (Optimizer)"
date: 2026-05-27
draft: true
math: true
tags: []
categories: ["Fundamentals"]
description: ""
---

**This post currently contains known errors and will be corrected.**

## Where the nanochat use Nesterov
`muon_step_fused()` function in `optim.py`

## 0. Motivation
We reflect the momentum on the final update value $u$ in $\theta_{t+1} = \theta_t - u$.
In this situation, reflecting the gradient not simultaneously with momentum but after momentum reflection could be better because comparing to gradient+momentum update, applying momentum itself has similar direction anyway. So in this situation with one more step of updated location, probably the calculated gradient at that momentum based updated point indicate more direct loss decreasing direction.
So the order of update is different with standard momentum.
- Standard momentum: momentum+gradient at that point at the same time
- Nesterov momentum: there are two steps in update procedure
	1. momentum based parameter update is done first, then the parameter is moved. 
	2. At the moved parameter, the gradient is obtained, then we update the parameter with that obtained gradient.

## 1. Explanation
The basic principle and idea is similar to standard momentum but the order of update is different. Here's the update rule.
- $v_{t+1} = u*v_t + lr*g$, $g$ at $\theta_t + u*v_t$
	- $v_t$: Current momentum
	- $\theta_t$: Current parameter
	- $u*v_t$ term: First movement by the momentum
	- $lr*g$, $g$ at $\theta_t + u*v_t$: Second movement by the gradient at $\theta_t + u*v_t$
- $\theta_{t+1} = \theta_t - v_{t+1}$

**Time step of momentum**
Both standard and nesterov has same $v_t$ update equation of $v_{t+1} = u*v_t + lr*g_t$.
- $v_{t-1}$ becomes $v_t$ after calculating $v_{t-1}$ and $g_t$.
- $v_{t}$ and $g_t$ are used on $\theta_{t+1}$ update
- $v_t$ update is happened with $u*v_{t-1}$(prev v), and the $g_t$ at $u*v_{t+1}$ moved point
	- One update of $v_t$ step is $u*v_{t-1}$ and the g that's dependent on or moved by the $u*v_{t-1}$
- For the $\theta_{t+1}$ update, $g_t$ and $v_t$ are used

In the implementation, we delay one step of update.
- Theoretical explanation: This is intuitive but hard to implement because we have to calculate gradient twice.
- Implementation: Suppose current parameter is $\theta_t$ = $\theta_{t-1} + mv_{t-1}$  One update step is (1) Update parameter with gradient at $\theta_t$ (2) Update parameter with momentum of $v_t = u*v_{t-1} + lr*g_t$ for the next update beforehand 

## 2. Implementation

### 2-1. Standard way
```
grad = lr*grad
momentum = mu*momentum + grad # Suppose momentum in mu*momentum is previous one
grad.add_(momentum, alpha=mu)
```
= $\theta_{t+1} = \theta_t - ((lr * g_t) + u(u*v_{t-1} + lr*g_t))$

### 2-2. Implementation in pytorch SGD
They moved the position of lr
```
momentum = mu*momentum + grad # Suppose momentum in mu*momentum is previous one
grad.add_(momentum, alpha=mu)
grad.mul_(lr)
```
- https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/sgd.py#L28

$= \theta_{t+1} = \theta_t - (lr)*(g_t) - lr*u*v_t = \theta_t - lr*(g_t + u(u*v_{t-1} + g_t))$

### 2-3. Implementation in Muon
```
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta)
    # lr is skipped in here, lr would be applied at the last calculation of Muon
```
$= \theta_{t+1} = \theta_t - (1-b)*(g_t) - b*u*v_t = \theta_t - ( (1-b)*(g_t) + b*( b*v_{t-1} + (1-b)*g_t ) )$

## Compare Muon nesterov with pytorch SGD one
The below equation is not fully unrolled format, but quiet intuitive to compare them.
- Nesterov in pytorch: $\theta_{t+1} = \theta_t - (lr)*(lr * g) - lr*u*v_t$
- Nesterov in Muon: $\theta_{t+1} = \theta_t - (1-b)*(lr * g) - b*u*v_t$

The difference is the ratio between momentum and gradient.
We can control the ratio to move with gradient and momentum with beta.

## Reference
- https://tensorflow.blog/2017/03/22/momentum-nesterov-momentum/
	- Theoretical explanation of nesterov
	- The figure is really helpful to track the update process of nesterov momentum
- https://docs.pytorch.org/docs/2.12/generated/torch.optim.SGD.html
	- Pytorch nesterov implementation docs
	- Refer to "The implementation of SGD with Momentum/Nesterov subtly differs from Sutskever et al. and implementations in some other frameworks."
	- ```
	  v_{t+1} = μ∗v_t + g_{t+1}
	  p_{t+1} = p_t − lr∗v_{t+1}
	  ```
- https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/sgd.py#L28
	- Implementation of nesterov in pytorch
