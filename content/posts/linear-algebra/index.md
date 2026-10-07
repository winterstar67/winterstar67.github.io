---
title: "Linear Algebra"
date: 2026-05-19
draft: false
math: true
tags: ["Background"]
categories: ["Fundamentals"]
description: ""
---

# 1. Important theorem

## 1-1. Spectral Theorem
For any real symmetric matrix A, it has a basis composed of orthonormal eigenvectors
$A=QΛQ^T$
- $Q$: orthogonal matrix (all of the column vectors are orthonormal eigenvectors)
- $\Lambda$: diagonal matrix (eigenvalues)

If we apply multiplication of $A = QΛQ^T$ on input vector x, the effects are the following.
1. $Q^T$: Rotating the vector x with the rotation matrix $Q^T$
2. $\Lambda$: Scaling each rotated axis with $\Lambda$
3. $Q$: Invert the rotation of $Q^T$ to restore the original axis

# 2. Linear Transformation
Transformations of vector x by matrix A $y=Ax$ can be one of the following:
- Rotation
- Stretch(Scaling)
- Reflection
- shear
- projection

Every linear transformation can be represented with `rotation + scaling + rotation`, because every matrix A can be decomposed into U(rotation), $\Sigma$(scaling), and V(rotation)
So, $y = Ax = U\Sigma V^T x$ basically means applying rotation, scaling, and rotation

## 2-1. Scaling - A is diagonal matrix
It's easy to imagine: just map each element in the row vector x to each column in matrix A

## 2-2. Rotation
If we multiply A on x like `y=Ax`, this means that we changed from the coordinate system to another coordinate system.

### 2D rotation matrix
In the 2-dimensional space, the matrix A that rotates the vector with $\theta$ angle is
$A = \begin{bmatrix}  \cos{\theta} & -\sin{\theta}\\  \sin{\theta} & \cos{\theta}  \end{bmatrix}$
If we multiply matrix A onto the vector $x = \begin{bmatrix}  x\\  y  \end{bmatrix}$, it would be
$y = Ax = \begin{bmatrix}  x\cos{\theta} -y\sin{\theta}\\  x\sin{\theta} + y\cos{\theta}  \end{bmatrix}$.
This y is the vector obtained by rotating the vector x by angle $\theta$.

### Why the y vector is the same as x rotated with $\theta$ angle
Let's represent the vector x in polar form.
- $x = \begin{bmatrix}  r\cos{\phi}\\  r\sin{\phi}  \end{bmatrix}$, In here $x = r*\cos{\phi}$ and $y=r*\sin{\phi}$
If we rotate the angle from $\phi$ to $\phi + \theta$, the rotated vector would be 
- $x' = \begin{bmatrix}  r\cos{(\phi+\theta)}\\  r\sin{(\phi+\theta)}  \end{bmatrix}$
By the trigonometric addition formulas, 
- $\cos{(\phi+\theta)} = \cos{\phi}\cos{\theta} - \sin{\phi}\sin{\theta}$
- $\sin{(\phi+\theta)} = \sin{\phi}\cos{\theta} + \cos{\phi}\sin{\theta}$

So
- $x' = \begin{bmatrix}  r(\cos{\phi}\cos{\theta} - \sin{\phi}\sin{\theta})\\  r(\sin{\phi}\cos{\theta} + \cos{\phi}\sin{\theta})  \end{bmatrix} = \begin{bmatrix}  x\cos{\theta} - y\sin{\theta}\\  x\sin{\theta} + y\cos{\theta}  \end{bmatrix}$

It's exactly the same as $Ax$

### What matrix is the rotation matrix?
If the following conditions are satisfied, then it's a rotation matrix
1. orthogonal: $R^T R=I$ - Keeps the same length and angle of the vector
	- Same length: $||Rx||=||x||$ because $||Rx||^2 = (Rx)^T (Rx)=x^TR^TRx=x^Tx$ 
	- Same angle: $(Rx) \cdot (Ry) = (Rx)^T(Ry) = x^TR^TRy = x^Ty$ before and after applying R, they have the same length and the same angle.
	- An orthogonal matrix doesn't change the space - it just rotates or reflects it.
2. determinant: $\det{R}=1$
	- The property of trigonometric functions: $(\cos{\theta})^2 + (\sin{\theta})^2 = 1$

## 2-3. Reflection
If $A = \begin{bmatrix}  1 & 0\\  0 & -1  \end{bmatrix}$ and multiply it with x, then Ax would be $Ax = \begin{bmatrix}  x\\  -y  \end{bmatrix}$
- It's reflected by x-axis

## 2-4. The property of Rotation and Reflection
Both matrices are orthogonal matrices
- $A^T A = I$
- If we multiply the transposed matrix $A^T$ onto the rotation or reflection matrix $A$ applied vector, then the vector would be back to the original one


# 3. Eigen decomposition
The **eigenvector** of matrix A is the vector that doesn't change the direction but only the length in $Av=\lambda v$, and the **eigenvalue** is the scalar value that corresponds to the effect of multiplying $A$

## 3-1. Special case: Covariance matrix
In covariance matrix, the eigenvector has more meaning than keeping direction.
Each eigenvector means the direction along which the data variance is maximized.

# 4. SVD(Singular Vector Decomposition)
Any arbitrary matrix W can be decomposed in the following form
- $W = U \Sigma V^T$
- $WV = U \Sigma$
	- U: Left singular vectors
		- It's Orthogonal matrix
		- `singular vector` of U: The column of U
		- `singular direction`: The direction that the singular vector indicates
	- $\Sigma$: singular values
		- It's Diagonal matrix
	- V: Right singular vectors
		- It's Orthogonal matrix
		- `singular vector` of V: The column of V
		- `singular direction`: The direction that the singular vector indicates

For each single vector in $U$ and $V$, this can be written as $Wv_i = \sigma_i u_i$
- Here, just the definition of $u_i$ is $\frac{Wv_i}{\sigma_i}$
- The property of orthogonal matrix (U and V):
	- Columns in the matrix are independent of each other
	- $V^T V$ = $U^T U$ = $I$

Each term in SVD is a rank-1 matrix
- In that one rank-1 matrix, each column and row are linearly dependent on each other

We can interpret each orthogonal matrix as one new axis like the x, y, and z-axes ($[1,0,0], [0,1,0]$, and $[0,0,1]$ each) - in this view, we can interpret that even if we change one orthogonal rank-1 matrix(or axis), it doesn't affect the other axes(rank-1 matrix component)


What's the advantage of SVD decomposition?
- It can be used for approximation 
	- we can drop the vectors that have a low singular value
		- The reduced computation cost compared to computing the full matrix could be huge
- We can utilize the properties of orthogonal matrix and diagonal matrix in U, $\Sigma$, and V

# 5. Projection
The dot product of a unit vector and a vector x indicates the projection of vector x onto the unit vector direction.
