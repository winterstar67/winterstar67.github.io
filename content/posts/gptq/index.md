---
title: "GPTQ"
date: 2026-08-30
draft: false
math: true
tags: ["Paper"]
categories: ["Fundamentals"]
description: ""
---

All codes and results are in Github

# Paper info
- **Title**: *GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS*
- **Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh
- **URL** : https://arxiv.org/pdf/2210.17323
- **Length**: 16 pages

# 0. Background
1. Linear algebra
	- Cholesky decomposition
		- It's known as $A=LL^T$ decomposition too.
	- Schur complement
		- In an $n$ x $n$ square matrix A $$A = \left(\begin{array}{cccc}A_{11} & A_{12} \\ A_{21} & A_{22} \\\end{array}\right)$$, Schur complement is $A/A_{11} = A_{22} - A_{21} A^{-1}_{11} A_{12}$
			- $A_{11}$ - $A_{22}$: Block matrices
	- Inverse block matrix
		- The inverse of an $n$ x $n$ square matrix $A = \left(\begin{array}{cccc}A_{11} & A_{12} \\ A_{21} & A_{22} \\\end{array}\right)$ is $$A^{-1} = \left(\begin{array}{cccc}A_{11}^{-1} + A_{11}^{-1}A_{12}S^{-1}A_{21}A_{11}^{-1} & -A_{11}^{-1}A_{12}S^{-1} \\ -S^{-1}A_{21}A_{11}^{-1} & S^{-1} \\\end{array}\right)$$ where $S = A/A_{11} = A_{22} - A_{21} A^{-1}_{11} A_{12}$
	- Eigenvalues of the inverse of a matrix A 
		- If matrix $A$ is diagonalizable to be $A = Q\Lambda Q^T$, the inverse of matrix $A$, $A^{-1}$ is diagonalizable into $A^{-1} = Q\Lambda^{-1}Q^T$. Here, $\Lambda^{-1}$ has reciprocal elements of $\Lambda$.
			- $\Lambda$: Eigenvalues
			- $Q$: Eigenvectors
	- The $\text{trace}(A)$ is the same as the sum of eigenvalues
	- Through the cyclic property: $\text{trace}(ABC) = \text{trace}(BCA) = \text{trace}(CAB)$, the below result is derived. $Σ(\text{eigenvalues})$ is used in the dampening step
		```
		  trace(A) = trace(QΛQᵀ) = trace(ΛQᵀQ) eigenvalue decomposition
         = trace(Λ·I) (Q is orthogonal QᵀQ = I)
         = trace(Λ)
         = Σ(eigenvalues)
		```
2. Method of Lagrangian multiplier {{< wikilink "method-of-lagrangian-multiplier" >}}
3. Optimal Brain Quantization (OBQ) generalized Optimal Brain Surgeon (OBS)'s pruning framework to quantization.

# 1. GPTQ Explanation
The purpose of quantization is **minimizing activation error between applying/not applying the quantization on weights, which is $L(e) = ||WX - \hat{W}X||^2 = ||(W-\hat{W})X||^2 = ||eX||^2$**.
- $\hat{W}$: Quantized $d_{row}$ x $d_{col}$ weight matrix
	- Row: All inputs to one output
	- Column: One input to all outputs 
- X: $d_{col}$ x $n$ Input matrix
	- Row: All batch and one feature
	- Column: One data and all features
- $e=w-\hat{w}$ ($1\times d_{col}$)
	- Although I wrote $||(W-\hat{W})X||^2 = ||eX||^2$, I'll treat $e$ as a vector.
	- $w$: one of the $W$'s rows ($1\times d_{col}$)
- Compensating concepts: The quantization is not done at once. When there's a weight matrix, each element (weight) is quantized sequentially.
- After the first weight element is quantized, $||WX - \hat{W}X||^2$ incurs an error, and the other unquantized weights are adjusted to compensate for it.

## 1-1. Background concepts
1. OBQ uses a greedy order of weight quantization, but using a greedy order versus simply fixing the order by row has little effect on performance.
2. Ordering by a row gives the advantage that the inverse of the Hessian isn't recalculated per weight quantization. - GPTQ uses ordering by row

## 1-2. The order of understanding GPTQ
1. Get the approximation of $||WX - \hat{W}X||^2$ in Taylor expansion, $L(e) = L(0) + eg^{\top} + \frac{1}{2}eHe^T = 0 + 0 + \frac{1}{2}eHe^T = \frac{1}{2}eHe^T$.
	- Because the model training is already done, $WX$ is similar to the label data. So $WX$ is at global minima and gradient at global minima is 0.
	- The Hessian is dominant.
2. Second derivative of L(e): $H = L''(e) = \frac{2eXX^T}{de} = 2XX^T \rightarrow 2XX^T + \lambda I$ (Dampening), $\lambda = 0.01 * \text{mean}(\text{diag}(H))$ 
	- First derivative of L(e): $g = L'(e)= \frac{d((eX)(eX)^T)}{de} = 2eXX^T$
	- **Dampening**: Suppose that one of the eigenvalues of Hessian matrix $H$ is too low. Then, one of the eigenvalues of the $H^{-1}$ would be too high. This value would cause the update value $e^*$ too high. To prevent this, the dampening is adopted.
3. Through quantization, the error at position $q$ is now determined to be $e_q = \Delta_q$.
	- $e_q = e \cdot u_q$
	- $u_q$: One-hot vector where the $q$-th component is activated ($d_{col}\times1$ shape)
4. Under the $e_q = \Delta_q$ constraint, the purpose is minimizing $||WX - \hat{W}X||^2$. This minimization can be solved with the method of Lagrangian multiplier. The equation is $\mathcal{L}(e, \lambda) = ||WX - \hat{W}X||^2 + \lambda(e \cdot u_q -\Delta_q) = \frac{1}{2} eHe^T + \lambda(e \cdot u_q-\Delta_q)$. 
	- $\nabla_e \mathcal{L} = H e^T + \lambda u_q = 0 \rightarrow e = -\lambda u_q^T H^{-1}$. Substituting this $e$ into $eu_q = \Delta_q$ then $\lambda = -\frac{\Delta_q}{[H^{-1}]_{qq}}$
		- $[H^{-1}_{qq}] = H^{-1}[q,q] = u_q^T H^{-1} u_q$
	- Now substitute this into  $e = -\lambda u_q^TH^{-1}$ then $e^* = \frac{\Delta_q}{[H^{-1}]_{qq}} u_q^T H^{-1}$
		- $u_q^T H^{-1} = H^{-1}_{q,:}$
		- **In the implementation of GPTQ, $\Delta_q$ would be a vector and $\Delta_q$ and $H^{-1}_{q,:}$ should be calculated by outer product because each row of $\Delta_q$ has the operation with the same $H^{-1}_{q,:}$, since $H^{-1}$ is only dependent on 2XX^T**
5. This $e^{*}$ is used for the update: Quantizing $w_q$ and slightly adjusting the other weights $w_i$s for the compensation.
Because $e = W - \hat{W} \rightarrow \hat{W} = W - e$, we subtract $e^*$ from the corresponding row of $W$.

But there's a floating-point error issue. When $H^{-1}_{i}$ is calculated recursively in $H^{-1}_{1} \rightarrow H^{-1}_{2} \rightarrow H^{-1}_{3} \rightarrow ...$ order, the floating-point error would be accumulated.
So, to make $H_{i+1}$ not depend on $H_{i}$ but be derived from the initial inverse of the Hessian, matrix $H^{-1}_{1}$, the authors used *Cholesky*($LL^T$) decomposition.

### 1-2-1. How the Cholesky decomposition solves the accumulation of floating-point error
#### 1-2-1-1. $H^{-1}_{i+1}$ obtained naively from the $H^{-1}_{i}$ case
We saw that $H^{-1}_{i+1}$ has accumulated floating-point error that $H^{-1}_{i}$, $H^{-1}_{i-1}$, and $H^{-1}_{i-2}$, ... calculations caused.
Suppose $M = H^{-1} = \left(\begin{array}{cccc}a & b^T \\ b & C \\\end{array}\right)$. Then, by the formula for the inverse of a block matrix, $(M^{-1})_{22} = (C-ba^{-1}b^T)^{-1}$. Because $(M^{-1})_{22}=H_{2}$, $H_{2}^{-1}=C-ba^{-1}b^T$.
The other indices of the matrix are obtained in this way.

#### 1-2-1-2. $H^{-1}_{i+1}$ obtained from the Cholesky decomposition case
Suppose $A_1 = H^{-1}_1$ and $A_1=A = \left(\begin{array}{cccc}a & b^T \\ b & C \\\end{array}\right) = LL^T = R^TR$. Then $A_1 = R^TR = \left(\begin{array}{cccc}r_{11} & 0 \\ r_{12}^T & R_{22}^T \\\end{array}\right) \left(\begin{array}{cccc}r_{11} & r_{12} \\ 0 & R_{22} \\\end{array}\right)$.
- $a$: It's a scalar. $a=r_{11}^2$
- $b$: It's a column vector, $b=r_{12}^Tr_{11}$
- $C$: It's a block matrix, $r_{12}^Tr_{12} + R_{22}^TR_{22}$
- $L$: Component of Cholesky decomposition
- $R$: $L^T = \left(\begin{array}{cccc}r_{11} & r_{12} \\ 0 & R_{22} \\\end{array}\right)$
Then, 
1. $A_2 = H_2^{-1} = C-\frac{bb^T}{a}$
2. $\frac{bb^T}{a} = \frac{(r_{12}^Tr_{11})(r_{12}^Tr_{11})^T}{r_{11}^2} = r_{12}^Tr_{12}$
So, $A_2 = H_2^{-1} = r_{12}^Tr_{12} + R_{22}^TR_{22} - r_{12}^Tr_{12} = R_{22}^TR_{22}$.
This means that once the $R^TR$ is obtained at initial state, $A_1$, the squared right-bottom $R_{ii}$ matrix is utilized.

**But**, there's a simpler, more efficient way to get $H_{2}^{-1}$ than calculating $R_{22}^TR_{22}$.
Suppose $H^{-1} = \left(\begin{array}{cccc}H^{-1}_{11} & H^{-1}_{12} \\ H^{-1}_{21} & H^{-1}_{22} \\\end{array}\right) = R^TR$. Then it's $\left(\begin{array}{cccc}r^2_{11} & r_{11}r_{12} \\ r_{11}r^T_{12} & r_{12}^Tr_{12} + R_{22}^TR_{22} \\\end{array}\right)$.
Because we already have $r_{12}^Tr_{12} + R_{22}^TR_{22}$ which is $H^{-1}[2,2]$, $H_2^{-1}$ is obtained by $H^{-1}_{22}-r_{12}^Tr_{12} = H^{-1}_{22}-R[1,2]^TR[1,2]$.

**Also**, we don't even need to calculate $H^{-1}_{2}-R[1,2]^TR[1,2]$ because what we need from $H_{2}^{-1}$ are only the $H_{2}^{-1}[1,1]$ and $H_{2}^{-1}[:,1]$.
The formula below shows that we need only the first row of $R$.
Suppose $R=\left(\begin{array}{cccc}r_{11} & r_{12} & r_{13} \\ 0 & r_{22} & r_{23} \\ 0 & 0 & r_{33} \\\end{array}\right)$ and $H_{2}^{-1} = R^TR = \left(\begin{array}{cccc}r^2_{11} & r_{11}r_{12} & r_{11}r_{13} \\ r_{11}r_{12} & r_{12}^2+r^2_{22} & r_{12}r_{13}+r_{22}r_{23} \\ r_{11}r_{13} & r_{13}r_{12} + r_{23}r_{22} & r_{13}^2 + r_{23}^2 + r_{33}^2 \\\end{array}\right)$.
- $H_{2}^{-1}[1,1]$: It's $r_{11}^2$
- $H_{2}^{-1}[:,1]$: It's $r_{11} * R[1, :]^T$

# 2. Flow of the calculation
The steps of GPTQ are the following
```
[calibration data] → X1 (Input) obtained
                ↓
        Calculation of H1 from X1 → Quantization of W1 by column by column from left to right (Apply e* on W1 update)
                ↓ Layer 1 Done
        [calibration data] → (Quantized W1) → X2 obtained
                ↓
        Calculation of H2 from X2 → Quantization of W2
                ↓ Layer 2 Done
        [calibration data] → (Quantized W1, W2) → X3 obtained
                ↓
        ... (Repeat until every layer is quantized)
```

## 2-1. Example of the calculation steps
Suppose there are three layers, $W_1$, $W_2$, and $W_3$. I'll describe how the weights are changed.
- $W_{i} = \left(\begin{array}{cccc}w_{i1} & w_{i2} & w_{i3} & ... \\\end{array}\right)$
	- $w_{ij}$: A column vector of $j$-th weight in the $i$-th layer. **This $w$ is not the $w$ in Section 1. They are unrelated.**
### 2-1-1. One weight update (Inner loop of $W_1$)
The input activation of $W_1$, $X_1$, is obtained
#### 2-1-1-1. Step 1
The first column of $W_1$ is updated using $X_1$.
$W_{1} = \left(\begin{array}{cccc}w'_{11} & w_{12} & w_{13} & ... \\\end{array}\right)$
- $w_{ij}$: The unquantized weight column
- $w'_{ij}$: The quantized weight column
#### 2-1-1-2. Step 2
The second column of $W_1$ is updated.
$W_{1} = \left(\begin{array}{cccc}w'_{11} & w'_{12} & w_{13} & ... \\\end{array}\right)$
#### 2-1-1-3. Step 3
The third column of $W_1$ is updated.
$W_{1} = \left(\begin{array}{cccc}w'_{11} & w'_{12} & w'_{13} & ... \\\end{array}\right)$

### 2-1-2. Several weights update (Outer loop)
#### 2-1-2-1. Step 1 (Layer 1 update)
The input activation of $W_1$, $X_1$, is obtained
The whole $W_1$ is updated using $X_1$.
$W'_{1} = \left(\begin{array}{cccc}w'_{11} & w'_{12} & w'_{13} & ... \\\end{array}\right)$
- $W'_1$: The quantized weight (layer 1)
- $w'_{1j}$: The quantized weight column in the layer 1

#### 2-1-2-2. Step 2 (Layer 2 update)
The input activation of $W_2$, $X_2$, is obtained after $W_1$ quantization is done.
The whole $W_2$ is updated using $X_2$.
$W'_{2} = \left(\begin{array}{cccc}w'_{21} & w'_{22} & w'_{23} & ... \\\end{array}\right)$
- $W'_2$: The quantized weight (layer 2)
- $w'_{2j}$: The quantized weight column in the layer 2

...
