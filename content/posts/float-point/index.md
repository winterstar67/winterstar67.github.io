---
title: "Float Point"
date: 2026-06-15
draft: false
math: true
tags: ["Background"]
categories: ["Fundamentals"]
description: ""
---

# Float Point
It's the method to represent the decimal in binary notation.

# 1. The composition of float point
Float point is composed of (1) sign bit (1-bit), (2) exponent bit (8-bit), (3) significand (mantissa)

## 1-1. Sign
This means the sign of the number, + or -
 - 0 means positive
 - 1 means negative

## 1-2. Exponent
What's the maximum value that the fp can express? 
What's the limit of range of the fp value?
- The exponent 8-bit doesn't mean $2^8$. 8 bit is the num of possible cases which is 256 
	- so you should think as $2^{00000000_2 - 127}$ \~ $2^{11111111_2 - 127}$ way.

## 1-3. significand (mantissa)
How many steps can we divide the number into? 
- In the case of 1.11111, it can be divided in $2^{-5}$ precision. 1.1111111111 can be divided in $2^{-10}$ precision
But it doesn't always have $2^{-10}$ even when you have 10 bits for mantissa. You should think about the exponent part ($2^{e-127}$) together.
Let's say the exponent is $2^2$ . Then it would be $1.1111111111_2 * 2^2 = 111.11111111_2$ . The smallest step part now becomes $2^{-8}$. So the smallest precision step size differs depending on the exponent.

In $e-127$, the 127 is the bias. If we just use $2^e$ (e = $[0, 255]$) , then we can't express the value $[0, 1)$ part. So we set bias to be 127.

Just think the precision part ($.010101010$ which is after point part in $1.010101010$) like this. Would you like to add or turn the number in $2^{-2}$ position in $1.xxxxxx$? If we add it, 1 would be added to the second $x$ which is in $.xxxxxx$. So, if it was $1.x0xxxx$, it would be $1.x1xxxx$ after adding $1$.
$1.111111$ is the same as $2^0+2^{-1}+2^{-2}+2^{-3}+2^{-4}+2^{-5}+2^{-6}$

Now with this principle. We can assume when we encounter the precision or handle the float point or need to decide which fp to use among fp32, fp16, or bf16
- Actually, it's hard unless you debug the gradient or forward pass a lot. But, at least you can predict the impact of choosing a certain float point now!

# 2. Example of converting the decimal number into binary expression
## 2-1. Set the number
Suppose we have a number $9432.302193$, and we will convert it into binary expression.

## 2-2. Convertion
- Integer part: $9432 = 10010011011000_2$
- Decimal part: $.302193 = .01001101..._2$
- Total converted result: $10010011011000.01001101..._2$

## 2-3. Normalization (Moving decimal point)
To leave the only $1$ in the integer part, we should move the decimal point + 13 into the left direction.
So, $1.001001101100001001101... \times 2^{13}$
- If the given value is larger than 1, then we move the decimal point into left direction.
	- It's the situation that how large value can we express with Exponent bit(8 bit)?
- If the given value is smaller than 1, then we move the decimal point into right direction.
	- It's the situation that how much small value can we express with Exponent bit(8 bit)?
- Exponent is not about precision but just scale

## 2-4. Express it with f32
### 2-4-1. Sign bit (1-bit)
It's 0 because it's positive

### 2-4-2. Exponent bit (8-bits)
Store 13 because we moved 13 points in $1.001001101100001001101... \times 2^{13}$, but there's the bias of 127. So $13+127 = 140$ is stored in Exponent bit.
- 8-bits = $2^8$ is enough to store 140

### 2-4-3. Mantissa bit (23-bits)
23 numbers in $.001001101100001001101...$ part are stored in Mantissa bit
- Each $.00100$ bit corresponds to $2^{-1}, 2^{-2}, 2^{-3}, 2^{-4}, 2^{-5}, ...$

## (Optional) 2-5. How to get the true value from f32 expression (Restore)
Suppose we have $1.0101100101 \times 2^x$ expression. First, multiply $2^x$ by the `1.mantissa` bit part. And convert the resulting binary expression into the decimal expression.

# 3. The case of overflow and underflow by the limitation of Exponent
## 3-1. Overflow
Suppose there's $10^{1000000}$. Then, in the $xxxxx.yyyyy$ (x is the integer part, y is the decimal part), we don't have to care about y, because it's 0. So, what we care about is moving all x(in binary) into the decimal part. Then, in $1.yyyyy * 2^i$, $i$ would be larger than 128, which causes overflow in fp32

## 3-2. Underflow
Now suppose we have $10^{-1000000}$, Again among the $xxxxx.yyyyy$, we don't have to care about x but only y. Because it's too small, we have to move to left decimal and the binary expression would be $1.yyyyy * 2^{-i}$. $i$ would be too large so the exponent can't express it.
- If it's tooooo small like $10^{-1000000}$, then it would be $0.yyyyy$ unlike the overflow case. $1.yyyyy$ would not be possible because we can't reach to 1 in decimal numbers with the limited exponents
# 4. The limitation when Mantissa is too small
## 4-1. Rounding error
Suppose we have $10010011011000.01001101..._2 = 1.001001101100001001101... \times 2^{13}$.
But if we have f16 whose mantissa bit is 4. Then we can only store four decimal numbers which is $.0010$ in $.001001101100001001101...$ 
## 4-2. Loss of information
When we add $1,000,000 + 0.0001$, the $0.0001$ is out of precision of $1,000,000$. So, the addition result is the same as if nothing had been added. The result is the same as $1,000,000$ which is before performing the addition.

# 5. Calculation of float points (I copied this in Gemini chat)
## 5-1. Add
**The Steps:**
1. **Compare** the exponents to find the difference.
2. **Shift** the mantissa of the smaller number to the right by the difference in exponents. This aligns the binary points.
3. **Add** (or subtract) the mantissas.
4. **Normalize** the result. If adding causes the number to carry over (e.g., $10.xxx_2$), shift the mantissa right and increment the exponent.

**Example: $1.25 + 2.5$**
- **Step 1: Extract parts (including the hidden '1')**
    - $1.25 = 1.01_2 \times 2^0$
    - $2.5 = 1.01_2 \times 2^1$
- **Step 2: Align exponents**
    - $1.25$ has the smaller exponent ($0 < 1$). Shift its mantissa right by $1$ position.
    - $1.25 \rightarrow 0.101_2 \times 2^1$
- **Step 3: Add mantissas**
    - $0.101_2 + 1.010_2 = 1.111_2$
- **Step 4: Normalize**
    - The result is $1.111_2 \times 2^1$. It is already in the valid `1.xxx` format, so no further shifting is needed.
    - Converted to decimal, $1.111_2 \times 2^1 = 3.75$.

### 5-1-1. Type matching
In the case above, $1.01_2$ becomes  $1.010_2$ to match the length with $0.101_2$. 
This principle is applied on fp16 + fp8 operation too. fp8 becomes fp16 by expanding it with 0s
- I read about Quantization, and noticed that when we operate with quantized form, every operand should be quantized to get the benefit of operation speed increase.
	- If we quantize the weight but don't quantize the activation (inputs), there's no speed gain

### 5-1-2. A "Messy" Addition Example (Featuring $1+1$ and $0+0$)
Let's add **$3.5$** and **$5.75$**. This will trigger intense carry-overs.
- **Step 1: Extract Parts**
    - $3.5 = 11.1_2 = \mathbf{1.11_2 \times 2^1}$
    - $5.75 = 101.11_2 = \mathbf{1.0111_2 \times 2^2}$
        
- **Step 2: Align Exponents**
    - The smaller exponent is $2^1$ ($3.5$). We must shift it right by 1 to match $2^2$.
    - $1.11_2 \times 2^1 \rightarrow \mathbf{0.111_2 \times 2^2}$
        
- **Step 3: Add Mantissas (The diverse case)**
    - Let's pad $0.111_2$ with a zero so it matches the length of $1.0111_2$.
    
    ```
	Plaintext
       111   <-- (Carry row)
      0.1110 <-- (3.5 aligned)
    + 1.0111 <-- (5.75)
    --------
     10.0101
    ```

### 5-1-3. Principle of addition
The reason that when we add two 1s in the same position, the next position's digit has +1 is because it's binary.
Example: $2^x + 2^x = 2*2^x = 2^{x+1}$, which is the next number
- Like we do $13 + 17$ = 10 (by 3+7) + 10 (in 13) + 10 (in 17)

## 5-2. Multiplication
**The Principle: Add Exponents, Multiply Mantissas**

Multiplication is actually simpler than addition when it comes to exponents. Because $x^a \times x^b = x^{a+b}$, you can operate on the exponents and mantissas almost entirely independently.

**The Steps:**
1. **Add the exponents.** _(Hardware note: Because both exponents store the +127 bias, adding them together results in a double bias. The hardware subtracts the bias once to fix this)._
2. **Multiply the mantissas** (including the hidden `1.`).
3. **Determine the sign** using an XOR operation (if signs match, positive; if different, negative).
4. **Normalize** the result. Multiplying two numbers between $1.0$ and $1.999...$ can yield a number up to nearly $4.0$ ($11.xxx_2$). If this happens, shift the mantissa right by one and add $1$ to the exponent.
    
**Example: $1.25 \times 2.5$**
- **Step 1: Extract parts**
    - $1.25 = 1.01_2 \times 2^0$
    - $2.5 = 1.01_2 \times 2^1$

- **Step 2: Add exponents**
    - $0 + 1 = 1$

- **Step 3: Multiply mantissas**
    - $1.01_2 \times 1.01_2$
    - In decimal math, this is $1.25 \times 1.25 = 1.5625$. In binary, $1.5625$ is $1.1001_2$.
        
- **Step 4: Normalize**
    - The result is $1.1001_2 \times 2^1$. It starts with `1.`, so it is already normalized.
    - Converted to decimal, $1.1001_2 \times 2^1 = 3.125$.

### 5-2-1. A "Messy" Multiplication Example
Let's multiply **$3.5$** by **$3.5$**. This will show how binary long multiplication stacks up and requires normalization at the end.

- **Step 1: Extract Parts**
    - $3.5 = \mathbf{1.11_2 \times 2^1}$
    - $3.5 = \mathbf{1.11_2 \times 2^1}$

- **Step 2: Add Exponents**
    - $1 + 1 = \mathbf{2}$
        
- **Step 3: Multiply Mantissas**
    - We need to multiply $1.11_2 \times 1.11_2$. We do this just like regular long multiplication, ignoring the decimal point until the end.
    
    ```
    Plaintext
          111
        x 111
        -----
          111   <-- (111 * 1)
         1110   <-- (111 * 1, shifted left)
        11100   <-- (111 * 1, shifted left twice)
        -----
       110001
    ```
    
    _(If you add those three rows up, you get several $1+1+1$ situations triggering heavy carry-overs!)_
    - Now, place the decimal point. Both original numbers had 2 digits after the decimal point (`.11`), so our result needs $2 + 2 = 4$ digits after the decimal.
    - Result: **$11.0001_2$**

- **Step 4: Normalize**
    - Currently, we have **$11.0001_2 \times 2^2$**.
    - It violates the `1.xxx` rule. We must shift the decimal left by 1 and bump the exponent up by 1.
    - Final normalized result: **$1.10001_2 \times 2^3$**

- **Verification:**
    - $1.10001_2 \times 2^3 = 1100.01_2$
    - $1100.01_2 = 8 + 4 + 0.25 = \mathbf{12.25}$. ($3.5 \times 3.5 = 12.25$).

### 5-2-2. Principle of multiplication
Suppose we are multiplying 111 * 111
For the second 1 in 111 in the later one (So it's x1x)
When we multiply 111 by that second 1. It would be 1110.
In the decimal system, the second 1 is 2^1 = 2 and 111 is 7.

$111_2 * 1x_2$ is the same as $(2^2 + 2^1 + 2^0) * 2^1 = 2^3 + 2^2 + 2^1$. Each digit is shifted to the left.
So,
```
  1.11
x 1.11
-----
0.0111 <-- (111 * 1, shifted right twice)
0.1110 <-- (111 * 1, shifted right)
1.1100 <-- (111 * 1)
-----
11.0001
```
Make sense.
Adding two exponents would be the same principle as multiplying 2^(x+y) so in terms of decimal system, 2^a becomes 2^(a+x+y) which shifts the binary to the left.
So it's $11.0001 * 2^2 = 1.10001 * 2^3$

# 6. Validate the logic
How to judge whether the principle makes sense is mapping the situation onto the decimal system.

# 7. About precision
Normally, the higher mantissa means the higher precision, but what if exponent bit is 1 while mantissa is high?
Then it can't represent things such as 0.05490354801 because the range doesn't reach to 0.0000000001.
But if the exponent is higher, then at least we can escape from 0 and can reach at some point while still its precision is not good.
So, for the approximated representation, the mind of "Mantissa == Precision" could be quite dangerous.
Two components need a certain num of bits.
