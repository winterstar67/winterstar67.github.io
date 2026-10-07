---
title: "Quantization"
date: 2026-06-27
draft: false
math: true
tags: ["Background", "Inference", "Efficiency"]
categories: ["Fundamentals"]
description: ""
---

# 0. Background
- {{< wikilink "float-point" >}}
- Stride: To understand `_to_col_major`, which makes the matmul operation efficient, stride concept is needed

# 1. Fast execution
Before the emergence of GPT, there have been demands on [the fast neural net training/inference](https://arxiv.org/pdf/1712.05877).
With the deployment and AI services such as ChatGPT, the importance of fast and efficient inference has got especially attention.
There are many techniques to realize this efficiency such as Quantization, Pruning, KV cache, and NPU, etc.
In this post, I'll organize Quantization that I've studied 

# 2. Quantization
**Quantization** is a technique that makes the model run faster by reducing bytes of data like changing dtype from large bytes (FP32/FP16) into the lower bytes (FP16/FP8).
The effects of decreasing the bytes of data, such as fp32 $\rightarrow$ fp8 (Quantization) are the following:
- Data going through memory bandwidth would be decrease by decreasing the bytes of data
- The lower bytes operation is faster than operation with higher bytes
- The Quantization and efficient inference are frequently mentioned in NPU industry
- In terms of Arithmetic intensity, quantization helps to decrease both `Compute bound` and `Memory bound`
### What bytes are decreased?
Quantization reduces the bytes of weights, data, and gradients by converting their dtype from FP32 into FP8. This makes running the model faster but the precision decreased.
- Example of precision decrease: activation value 0.3213123712 would be 0.321 
So after we get advantage of speed by quantized data operation, we need to restore (**dequantize**) the quantized one to the original scale using the scaler.

As a quantization target scale, INT8 is the commonly mentioned. But I'll describe FP8 quantization because nanochat used it.

### Summary of quantization steps
1. Choose between symmetric quantization and asymmetric quantization depending on the processing data
	- It decides the design of framework how to implement the quantization 
2. Get scaler `S = (x_min - x_max)/(q_min - q_max)` and $Z$
	- `x`: Original value
	- `q`: Quantized value
	- `Z`: Zero point, if you use symmetric quantization, you don't need to get Z. Z is computational overhead
		- The role of Zero point `Z` is to map the start point (min value) of original value into the start point (min value) of quantized value.
			- Suppose we only have scaler `S` but no `Z`. Then we only have an information of what's the ratio of change of original value against quantized value when we move quantized value to be +1?
			- Just knowing the ratio is meaningless because this ratio is constant in any of distance between two arbitrary data points.
			- But if we set the min value of original value to be mapped into min of quantized value with `Z`, we can know the original value can be mapped into quantized value within the quantization range
			The definition of symmetric quantization is `Z=0 => q_min == x_min/S`
3. Apply `S` on the original data with `q = round(x/S) + Z` equation
4. Do your calculation such as matmul in forward/backward with quantized value
5. After calculation, dequantize the values

## 2-1. Quantization Equation

### 2-1-1. Dot product situation
Suppose we'd like to quantize the $x$ and $w$ so that the dot product operation below becomes faster.
$$y = \sum_i{x_i * w_i}$$
- $x_i$: i-th input
- $w_i$: Weight multiplied with $x_i$
- $y$: Output

## 2-2. Scale calculation
1. $S$ is the scale of $x$.
$$S = \frac{x_{max}-x_{min}}{q_{max}-q_{min}}$$

	- $s_x$ scale can quantize the data and also restore the quanized data by $s_x$
	- It can be interpreted as a ratio between quantized value and original value. 
		- "When I move 1 in quantized scale, how much would the original value be changed?"
	- This is needed when we quantize and dequantize the values

2. (Optional) If you use asymmetric, you'll need Zero point $Z$.
$$ Z = q_{min} - \frac{x_{min}}{S}$$

3.  We can convert $x$ into the integer expression, $\tilde{x_i}$ with $$\tilde{x_i} = round(\frac{x}{s_x}) + Z$$
	- If you use symmetric quantization, Z = 0
4. We can restore quantized data $\tilde{x_i}$ into $x$ with $$x = \tilde{x_i} * s_x - Z * s_x$$
## 2-3. Dot product with quantized data
Original dot product equation: $$y = \sum_i{x_i * w_i}$$
Quantized dot product equation:
$$y = \sum_i{(\tilde{x_i}*s_x - Z_x*s_x) * (\tilde{w_i}*s_w - Z_w*s_w)} = s_x * s_w \sum_i{(\tilde{x_i}-Z_x) * (\tilde{w_i}-Z_w)}$$
The equation itself is the same as the original one, but the calculation efficiency becomes more efficient.
1. The bytes of $\tilde{x_i}$ and $\tilde{w_i}$ are lower than the original one. So loaded memory bandwidth data size is decreased.
2. Summation is done with decreased dtype data
3. (Symmetric case) If we use symmetric quantization, the alpha term would be removed because $Z = 0$. This is the reason for using the symmetric one rather than the asymmetric one when the symmetric one is allowable.

## 2-4. The range of scaler
There are two options for FP8 quantization.
### 2-4-1. e4m3
- Max value: 448
- It's used for **activations** and **weights**
### 2-4-2. e5m2
- Max value: 57344
- it's used for **gradients**

### 2-4-3. Decision between e4m3 and e5m2
If we can cover most of the values, then invest (allocate) the smallest bits on the exponents that are enough to cover the range and invest (allocate) the left bits on the mantissa.
- Because the gradient can be very large, we allocate one more bit on the exponent while the activations and weights are not usually large, so they have more mantissa (precision).

## 2-5. Options in Quantization

### 2-5-1. Static Quantization vs. Dynamic Quantization
This is about when to decide the mapping rule from original value to the quantized one
#### 2-5-1-1. Static quantization 
Static quantization is a method that makes a mapping rule before running the model. 
It's faster because once we set the rule of mapping from original value to quantization value, we don't have to calculate it again. But while running the model, we can't change the mapping rule. So It's vulnerable to underflow/overflow 
- All weight and activation values are mapped using a pre-allocation rule
- It's usually adopted in vision tasks.
#### 2-5-1-2. Dynamic quantization
Dynamic quantization is a method that maps the activation, weight, and data values while running the model.
It dynamically decides the mapping rule between the range of quantization and the range of original values. It has less risk of overflow and underflow, but every time we calculate the matmul, we should re-calculate the mapping range so it's slower than a static method
- In every operation where we use the quantization, we calculate the quantization range and mapping rule before quantizing them.
	- This ensures that all values are mapped within the max/min value of quantization.
- In LLM, it's dangerous to use static because the distribution of value in each position fluctuates more than that of vision features.
- In nanochat, **Dynamic quantization** is adopted

#### What should care in dynamic quantization
One of the important things in dynamic one is not to cause underflow/overflow problem. Mathematically, the equation would not cause the problem, but in practice, the float point operation can cause the underflow/overflow

### 2-5-2. Symmetric Quantization vs. Asymmetric Quantization
#### 2-5-2-1. Symmetric Quantization:
##### Pros
- `torch._scaled_mm` assumes that the Quantization is symmetric. The kernel doesn't have the argument of zero-point.
- There's no computational overhead which is caused by the Zero point term in the equation.
	- $-Z_a*\sum(q_b)$, $- Z_b*\sum(q_a)$, and $Z_a*Z_b*n$ terms are the overhead by the Zero point term.
- The mathematical equation and implementation are also simpler than the asymmetric one
	- We don't need to calculate Z
	- The abs of $x_{min}$ and $x_{max}$ in $[x_{min}, x_{max}]$ is the same. That's also equivalent to the $q_{min}$ and $q_{max}$ too.

##### Cons
* There could be a wasted range in quantized values
	* ReLU which has $[0, x]$ original range. If we map this into $[-127, 127]$ with Z=0, then $[-127, 0)$ is wasted because any original values are not being mapped to $[-127, 0]$.
	* The possibilities that can be mapped into the quantization range are just 128 which is $0, 1, 2, ..., 127$. When we dequantize(restore) the quantized value into original value, there could be duplicated values twice than the case that $[-127, 127]$ is fully used.
	* If we fully use $[-127, 127]$, then we can map the original values into total 256 numbers.

Use Symmetric Quantization when the data distribution is well distributed in the symmetric range. Then you can quantize it with computational efficiency

#### 2-5-2-2. Asymmetric Quantization:
This is the opposite of symmetric quantization. When you are not sure that your data distribution can be designed to be symmetric quantization (Or Z=0), then use this quantization running the risk of computational inefficiency.

# 3. Implementation
There are several cautions that are not considered in theoretical equations
1. Float Point 64
2. clamp
3. cast down

## 3-1. Quantization scaler
There are assumptions on obtaining the quantization scaler
1. It's symmetric quantization - which means we don't need zero point value
2. Dynamic quantization is used
In quantization itself, there's no heavy operation like matmul. So, for the accurate quantization, it's okay to use fp64 or fp32.

### 3-1-1. Quantization value range
`fp8_max = torch.finfo(fp8_dtype).max`
- We obtain the maximum value of fp8.
- This would be used to get quantization range. The range is `[-fp8_{max}, fp8_{max}]`.
- min fp8 is not needed because the quantization is symmetric. 

### 3-1-2. Original value range
`x_bound = x.abs().max()`
- `x.abs()` converts every element in a tensor to its absolute value. Then the maximum value of abs elements is obtained.
- By doing that, the original value range can be designed to cover all elements with `[-x_bound, x_bound]`
- Because the symmetric quantization is used, the range is not that tightly set. Because the distribution is symmetric, it's allowable.

### 3-1-3. Scaler
`S = ((x_bound.double())/(fp8_max)).to(torch.float32)`
- The original scaler equation is `S=(x_max - x_min)/(fp8_max-fp8_min) + Z`. But because it's symmetric, the implementation derives the same result.
	- `x_max == -x_min == max(x.max.abs() , x.min.abs())`
	- `fp8_max == -fp8_min == max(fp8.max.abs() , fp8.min.abs())`
	- `Z=0`, so we don't calculate the `Z`.
- `x_bound.double()` makes the dtype fp64.
	- We need fp64 in the division of `x_bound/fp8_max` because the error from float point operation can make the scaler different in `torch.compile mode` and `eager mode` 
	- If the division is done with fp64 and the result is converted into fp32, then whatever the mode is used, the result would be always same.
	- The operation reorder can produce a different result caused by the float point operation. And the torch.compile can reorder the operation order.
	  The reorder phenomenon is the same on the float64 too, but if the original maximum pool that we use is limited to float32, the error in the original scale
	  float32 can be prevented in float64 scale
	  ```
	  float64 eager:   0.00669642857 1...
	  float64 compile: 0.00669642857 2...  ← difference at 13th digit
	                         ↓ .float()
	  float32 both:    0.0066964          ← same, difference was invisible to
	  ```
  
- `.to(torch.float32)` makes the dtype fp32. This is needed because the `torch._scaled_mm` requires the scaler to be fp32.
	- I need reference source that the scaler should be fp32.

### 3-1-4. Quantization
`quantized_x = (x/(S.clamp(min=1e-10))).clamp(-1*fp8_max, fp8_max).to(fp8_dtype)`
- `S.clamp(min=1e-10)`: The clamp prevents the `x/0` case.
- `(x/(S.clamp(min=1e-10))).clamp(-1*fp8_max, fp8_max)`: There could be a case that `(x/(S.clamp(min=1e-10)))` ensures the quantized value stays within fp8 before casting it into fp8 dtype. Because it's the result of a **division operation**. So a slightly higher value can occur than the fp8 maximum value by float point error.
	- Because the fp32 becomes fp8, it ensures that the cast fp8 value is the same in both compile mode and eager mode.
- `to(fp8_dtype)` makes the dtype fp8. Before this casting, the dtype was fp32, just the value range was within fp8 max value

## 3-2. Forward
`_Float8Matmul` class inherits `torch.autograd.Function`.
- This enables the `_Float8Matmul` class to use `@staticmethods` in both `forward` and `backward` functions 

`def forward(ctx, input_2d, weight)`:
- `ctx`: It's a container that has some variables and functions.

### 3-2-1. Quantization of weight and data
```
quantized_input, S_input = _to_fp8(input_2d, torch.float8_e4m3fn)
quantized_weight, S_weight = _to_fp8(weight, torch.float8_e4m3fn)
```
### 3-2-2. FP8 specialized matmul
`result = torch._scaled_mm(quantized_input, _to_col_major(quantized_weight), scale_a=S_input, scale_b=S_weight, out_dtype=torch.bfloat16)`
- `torch._scaled_mm`: This is a matmul kernel for FP8 operands.
	- It supposes the quantization is symmetric which means there's no `Z` term.
	- The dtypes of `scale_a` and `scale_b` should be `torch.float32`
	- cuBLAS FP8 kernel (It exists only in H100) (@ is cuBLAS general GEMM kernel for BF16/FP32)
- `_to_col_major`: It makes the access on the column elements efficient by rearranging the memory arrangement.
	1. Suppose we are trying to reallocate the memory of elements in `X` in the `W@X` operation.
	2. The target to change memory arrangement is `X` and the shape of `X` is `[m,n]`. Then the stride of it is `[n, 1]`.
		- This means each column is continuous, and in row direction, the elements have a distance of `n` memory addresses
		- In the `W@X`, the memory access on `X` is carried out with the per-column unit.
		- So, memory access would be done with `n` unit like memory address `[0], [n], [2n], [3n]. ...` 
			- If n is 64, then the number of GPU's memory read is the number of rows in `X`.
	3. `_to_col_major` changes the stride to be `[1,n]`. Because the column unit in `X` participates in the dot product, reading the memory with column unit would be efficient. So it changes strides to be `[1,n]` in original shape `[m,n]`. 
		- It starts with `X` with shape `[m,n]` and stride `[n,1]` 
		- (1) First it does transpose, then `X.t` would be `[n,m]` and stride is `[1,n]`
		- (2) Reallocate the stride to be `[n,1]` by allocating new memory. Shape is `[n,m]` and stride is `[n,1]`
		- (3) transpose it again to have same tensor with original `X`. Shape is `[m,n]` and stride is `[1,n]`.
	- GPU reads data in the 64 bytes unit per one memory read. So, rearranging the memory order decides how many times GPU reads the memory
- `out_dtype=torch.bfloat16`: It returns the next activation (The result of matmul) with dtype BF16. I need to figure out why the activation should be bf16

### 3-2-3. Save information required for backpropagation
`ctx.save_for_backward(quantized_input, quantized_weight, S_input, S_weight)`
- It stores `quantized_input`, `quantized_weight`, `Scale of input`, and `Scale of weight`
- Because the forward and backward functions are disconnected, the local variables in forward function can't be reused in backward function. `ctx` solves this problem

## 3-3. Backward
The entire operations are the same as the forward function
- The concept of gradient of `matmul(matrix, matrix)` is required
