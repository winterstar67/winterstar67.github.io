---
title: "KV cache"
date: 2026-07-06
draft: false
math: true
tags: ["Efficiency", "Inference"]
categories: ["Fundamentals"]
description: ""
---

# 1. Inference of GPT without KV cache
Settings:
- shape of Q, K, V: `(Seq_len, N_heads, Head_dim)`
- Token: Each word is one token
- Input: "The time flow"

4th token prediction:
- Input: "The time flow"
- Embed Q, K, V of "The time flow".
	- Q = $q_1, q_2, q_3$
	- K = ($k_1, k_2, k_3$)
	- V = ($v_1, v_2, v_3$)
- Calculate attention among three tokens
	- $$a = \left(\begin{array}{ccc}soft(q_1^Tk_1) & 0 & 0 \\soft(q_2^Tk_1) & soft(q_2^Tk_2) & 0 \\soft(q_3^Tk_1) & soft(q_3^Tk_2) & soft(q_3^Tk_3)\\\end{array}\right)$$
- Calculate values with $aV$
- After going through every attention block, it passes the last output layer whose shape is `(N_heads * Head_dim, vocab_size)`.
- By sampling the token according to the probability of the last sequence among the vocab_size, the 4th token is determined.
- Suppose the 4th one is "is"

5th token prediction:
- Input: "The time flow is"
- $q_4, k_4$, and $v_4$, which are from token of "is", are calculated.
- Calculate attention among four tokens
	- $$a = \left(\begin{array}{cccc}soft(q_1^Tk_1) & 0 & 0 & 0 \\soft(q_2^Tk_1) & soft(q_2^Tk_2) & 0 & 0 \\soft(q_3^Tk_1) & soft(q_3^Tk_2) & soft(q_3^Tk_3) & 0 \\soft(q_4^Tk_1) & soft(q_4^Tk_2) & soft(q_4^Tk_3) & soft(q_4^Tk_4) \\\end{array}\right)$$
- For the 5th token prediction, we have to calculate every $q_1, q_2, q_3, q_4$, $k_1, k_2, k_3, k_4$, and $v_1, v_2, v_3, v_4$

Here, the inefficiency occurs due to the duplicated calculation of k and v in attention from the input layer to the output layer in every next token prediction.
The purpose of using KV cache is to resolve this inefficiency without affecting the existing attention value.

# 2. KV cache
The concept of KV cache is storing the already calculated K and V of tokens and reusing them on the next token prediction so that the calculation is decreased.

## 2-1. Inference steps based on KV cache
Settings:
- shape of Q, K, V: `(Seq_len, N_heads, Head_dim)`
- Token: Each word is one token
- Input: "The time flow"

4th token prediction (pre-fill stage):
- Input: "The time flow"
- Embed Q, K, V of "The time flow".
	- Q = $q_1, q_2, q_3$
	- K = ($k_1, k_2, k_3$)
	- V = ($v_1, v_2, v_3$)
	- K and V are stored
- Calculate attention among three tokens
	- $$a = \left(\begin{array}{ccc}soft(q_1^Tk_1) & 0 & 0 \\soft(q_2^Tk_1) & soft(q_2^Tk_2) & 0 \\soft(q_3^Tk_1) & soft(q_3^Tk_2) & soft(q_3^Tk_3)\\\end{array}\right)$$
- Calculate values with $aV$
- After going through every attention block, it passes the last output layer whose shape is `(N_heads * Head_dim, vocab_size)`.
- By sampling the token according to the probability of the last sequence among the vocab_size, the 4th token is determined.
- Suppose the 4th one is "is"

5th token prediction:
- Input: "The time flow is"
- $q_4, k_4$, and $v_4$, which are obtained from token of "is", are calculated.
	- This $k_4$ and $v_4$ would be stored in the KV cache too
- **KV cache is used**: the stored $k_1, k_2, k_3$ and $v_1, v_2, v_3$ are loaded
	- We can reuse this because the $k_1, k_2, k_3$ and $v_1, v_2, v_3$ values are not changed. If this were a bidirectional transformer, then the KV cache would be unavailable. But GPT masks the future tokens of each current token, so it works
- Calculate attention among four tokens
	- $$a = \left(\begin{array}{cccc}? & ? & ? & ? \\? & ? & ? & ? \\? & ? & ? & ? \\soft(q_4^Tk_1) & soft(q_4^Tk_2) & soft(q_4^Tk_3) & soft(q_4^Tk_4)\\\end{array}\right)$$
	- `?` means calculation is not necessary.
	- For the 5th token prediction, the query that we only need is the 4th one

#### KV cache removes the calculation for $q_1$ \~ $q_{T-1}$, $k_1$ \~ $k_{T-1}$, and $v_1$ \~ $v_{T-1}$
- O($T^2$) becomes O($T$) in the attention calculation
- When the length of the token sequence is larger, KV cache would be more effective

## 2-2. Could KV cache be used on model training?
No.
- Think of gradient flow. For the next token prediction in the last token position, all K and V of every token are related to predicting the result.
- Also if we train on the prev tokens continuously not only the last token, then also we need to train on the prev Q value too.
- We cache and load(reuse) the K and V because that's used for the next token prediction, which means that the cached K and V are included in gradient flow process (training) too.

# 3. The efficiency from KV cache
How much is it fast?
- Execution time test code: https://github.com/winterstar67/model-efficiency-lab/tree/main/KV%20Cache
- Execution time experiment: https://winterstar67.github.io/posts/kv-cache-implementation/
