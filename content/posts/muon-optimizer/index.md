---
title: "Muon Optimizer"
date: 2026-05-31
draft: false
math: true
tags: ["Paper", "Optimizer"]
categories: ["Fundamentals"]
description: ""
---

# Post (Paper) information
- **Title**: *Muon: An optimizer for hidden layers in neural networks*
- **Authors**: Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein
- **Year**: 2024
- **URL**: https://kellerjordan.github.io/posts/muon/#shampoo

# 1. Background
Low gradient doesn't mean that that's an unimportant direction or if we move with that calculated magnitude gradient, it would 100% appropriately and efficiently decrease the loss. Vice versa for the large gradient too. $\rightarrow$ {{< wikilink "weight-gradient-loss-update-step-size-and-hessian" >}}
- Gradient is just impact of weight about the loss at micro range of a point.
	- Low gradient means when we move that weight, then the low decrease in small scale. But it doesn't mean you should move in a small amount. We don't know.
	- The information to decide which step is appropriate to decrease loss is curvature. The weight scale, and gradient scale itself at one point are not that important in deciding the gradient step.
- In here,
	- Each singular value one(gradient strength per new axis) does not interfere with each other. So just setting 1 would not affect each other axis. 1s are just appropriate values. (If it wasn't an orthogonal matrix, setting 1 in one axis would affect the other axis so that axis would not be 1 anymore)

# 2. Explanation and why it works
The principle and its own property, I think that's similar to Shampoo, the orthogonality makes each gradient independent, so rescaling one does not affect the others.

Anyway, the purpose of Muon update is to change all of the singular values of the gradient matrix to be 1
- This means in terms of column vectors and row vectors of $G^TG$ and $GG^T$, it will rescale the gradient on the eigenvector directions $\rightarrow$ The same as Shampoo
- The important one is rescaling the components of the gradient in the eigenvector directions, and that rescaling would not affect rescaling on the other directions

## 2-1. NS iteration:
$$\begin{align*}
G’ &:= aG + b(GG^\top)G + c(GG^\top)^2G \\
&= (aI + b(GG^\top) + c(GG^\top)^2)G \\
&= (aI + bUS^2U^\top + cUS^4U^\top)USV^\top \\
&= U(aS + bS^3 + cS^5)V^\top
\end{align*}$$
The $G'$ update iteration is to set the singular values to be approximately 1. For every step of $G'$ update iteration
- $U$ and $V^T$ are the same
- The singular values become closer to 1.
What we update is only the singular values.

# 3. Code
I didn't implement it myself. I just analyzed the code
```
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update
```

Refer to `### 2-3. Implementation in Muon` in {{< wikilink "nesterov-momentum-optimizer" >}} about the `momentum.lerp_` and `grad.lerp(momentum, beta) if nesterov` part 

## 3-1. `momentum lerp` and `nesterov`
Refer to {{< wikilink "nesterov-momentum-optimizer" >}}


## 3-2. Computational efficiency
```
    if G.size(-2) > G.size(-1):
        X = X.mT
```
When we calculate the following equation, $$\begin{align*}
G’ &:= aG + b(GG^\top)G + c(GG^\top)^2G
\end{align*}$$If the num of rows( e.g. 4096 ) is more than the num of columns( e.g. 1024 ), then the result shape $GG^T$ would be larger. 
But also, we can convert the result to be `1024 x 1024` by calculating that with $G^T$ instead of $G$.
- SVD of $G^T$ is $(U\Sigma V^T)^T = (V \Sigma U^T)$

Here's the $G'^\top$ update
$$\begin{align*}
G'^\top &:= aG^\top + b(G^\top G)G^\top + c(G^\top G)^2G^\top \\
&= (aI + b(G^\top G) + c(G^\top G)^2)G^\top \\
&= (aI + bVS^2V^\top + cVS^4V^\top)VSU^\top \\
&= V(aS + bS^3 + cS^5)U^\top
\end{align*}$$
When we need the original $G'$, then we can restore it by transposing $G'^\top$. 
Now, we have two options to update $G'$. 
1. Update $G'$ using G without transposing it
2. Update $G'$ based on $G^\top$, (1) update $G'^\top$ first, (2) transpose $G'^\top$ to be $G'$ at last.

In the NS iteration code, choosing among the $G$ and $G^\top$ is important for computational efficiency. If $G^\top G$ shape is smaller than $GG^\top$, then updating with $G^\top G$ is efficient. And if $GG^\top$ shape is smaller than $G^\top G$, then update with $GG^\top$ is computationally efficient.
- `b * A + c * A @ A`: The shape of A differs by $GG^\top$ and $G^\top G$
- `B @ X`: The shape of B differs by $GG^\top$ and $G^\top G$

In the code, the parts that judge and handle which dimension is large between row dimension and column dimension are
```
    if G.size(-2) > G.size(-1):
        X = X.mT
```
and
```
    if G.size(-2) > G.size(-1):
        X = X.mT
```


## 3-3. Normalization of gradient matrix
`X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)` code takes the normalization role

### 3-3-1. Normalize what?
It normalize every singular value in $\Sigma$ of $X=U\Sigma V^\top$ to be within range $[0,1]$

### 3-3-2. Why should we do normalization before NS iteration?
Because the singular value has no limitation on its value.
- Even $\sigma_1=1000$ is possible in theory.
Claude Code said the large singular value can ruin the NS iteration. If that happens, the singular value could diverge instead of converging to 1 after NS iteration. 

So we normalize the Gradient matrix before we get into NS iteration.

### 3-3-3. Why does that code line normalize the singular value only?
Frobenius norm can be expressed in $|X|_F^2 = \text{tr}(X^T X)$ because all added elements are squared of elements themselves.
Apply this on $SVD(X) = U\Sigma V^\top$. Then
  $$|X|_F^2 = \text{tr}(X^T X) = \text{tr}(V\Sigma^T U^T \cdot U\Sigma V^T) = \text{tr}(V
  \Sigma^T\Sigma V^T)$$
And because $tr(ABC) = tr(CAB)$,
$$= \text{tr}(\Sigma^T\Sigma V^T V) = \text{tr}(\Sigma^T\Sigma) = \sum_i
  \sigma_i^2$$
So dividing matrix $X$ by $|X|_F$ is the same as dividing $X$ by the square root of the sum of squared singular values.
Now, $\frac{X}{|X|_F}$ indicates $U \frac{\Sigma}{|X|_F} V^T$, and $|X|_F \geq \max_i{\sigma_i}$. It ensures that every singular value is less than 1 after normalization with Frobenius norm.

## 3-4. Adjusting weight scale constantly
`update *= max(1, update.size(-2) / update.size(-1))**0.5` code has that role.
### 3-4-1. Frobenius norm after NS iteration
We can measure that when the singular values are all 1s, it's after NS iteration.
1. Frobenius norm is $\sqrt{\sum_i{\sigma_i^2}}$. After the NS iteration, all $\sigma$s are approximately 1. Then the Frobenius norm would be the square root of num of $\sigma_i$, which is $\sqrt{n}$ 
2. After NS direction, $G=U \Sigma V^\top$ becomes $G' = UV^\top$. 
	- $G'^\top G' = (UV^\top)^\top (UV^\top) = (VU^\top)(UV^\top) = VV^\top$. Because $V^\top V=I$ by the orthogonality, $V^\top =V^{-1}$. So $G'^\top G'=VV^\top=I$.
	  So all columns in $G'$ are orthogonal to each other.
	  $|UV^\top|^2_F$ = $\sum_i{\text{column vectors}_i * \text{elements}^2}$ = $1*$num of columns = $n$.
	  $|UV^\top|_F = \sqrt{n}$

### 3-4-2. Why should we set $\sqrt{n}$ not $\sqrt{m}$?
$|G'|_F$ is a square root of the sum of each squared elements. To get the average scale of each parameter, we need to divide $|G'|_F$ with total num of elements, which is $\frac{|G'|_F}{\sqrt{mn}}$.
- We should use RMS not average to measure the scale of each parameter, average cancels the positive number and negative number while the RMS doesn't

Then what's the RMS of $|G'|_F$? The average scale of parameters after NS iteration.
- Suppose the shape of G is $(m,n)$
- $\frac{|G'|_F}{\sqrt{mn}}$ = $\frac{\sqrt{n}}{\sqrt{mn}}$ = $\frac{1}{\sqrt{m}}$

There's no fixed answer, but think what's the reasonable value.
Suppose that $m>n$ in $(m,n)$ shape. Then RMS of parameters is dependent on row size of matrix. If (4096, 1024) is the shape, then RMS of $G'$ = $|G'|_F$ = $\frac{1}{\sqrt{4096}}$.
But in the weight matrix, one row(1024 length vector) has meaning for one output feature in forward pass because one row(whose elements are indexed along the column direction) means the weights from all inputs to the one output. The inputs are connected by a linear combination for one output feature, but not across the row direction(one column).
If the scale of parameters(RMS of parameters) is dependent on row scale, it can ruin the linear combination result by row weight vector decreasing the scale of output than expected one. 
- In an extreme case, as an example, suppose the matrix shape is $(10^6,1024)$. Then the linear combination of 1024 input features for one output feature in the next layer would be like $\frac{in_1}{\sqrt{10^6}} + \frac{in_2}{\sqrt{10^6}} + ... + \frac{in_{1024}}{\sqrt{10^6}}$. It's almost 0, so the outputs would be almost 0
	- Though the RMS doesn't mean every parameter is exactly $\frac{1}{RMS}$, we can roughly assume like that.

But, if we rescale it with $\frac{\sqrt{m}}{\sqrt{n}}$, now $G'$ = $|G'|_F$ = $\frac{1}{\sqrt{1024}}$, so $\frac{in_1}{\sqrt{1024}} + \frac{in_2}{\sqrt{1024}} + ... + \frac{in_{1024}}{\sqrt{1024}}$. Frobenius norm is constant.
- Even if the row is too large, it doesn't affect the linear combination weight(1/RMS). 
- If the size of the column is changed, then automatically the RMS value changes to match the input scale

So rescaling code line is necessary after NS iteration.

# 4. Linear Algebra

## 4-1. Frobenius norm
The way to measure the magnitude of matrix with the following equation
Frobenius norm of matrix $X$ = $|X|_F$ = $\sqrt{\sum_{i,j}{x_{ij}^2} }$
- If $X = [[3, 4],[0, 0]]$, then $|X|_F$ = $\sqrt{3^2 + 4^2 + 0^2 + 0^2} = \sqrt{9 + 16} = 5$
Frobenius norm can be expressed in $|X|_F^2 = \text{tr}(X^T X)$ too.

# 5. Question
## 5-1. Why does Muon use nesterov in that way?
Because it shows better results

>Another purely empirical result is that using Nesterov-style momentum for Muon works a bit better than normal SGD-momentum in every case we have tested. We have therefore made this the default in the public [Muon implementation](https://github.com/KellerJordan/Muon).
- https://kellerjordan.github.io/posts/muon/#shampoo
