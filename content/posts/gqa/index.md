---
title: "GQA"
date: 2026-03-29
draft: false
math: true
tags: ["Paper", "Efficiency"]
categories: ["Fundamentals"]
description: ""
---

# Paper info
- **Title**: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- **Authors**: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai
- **Venue**: EMNLP 2023
- **URL**: https://arxiv.org/pdf/2305.13245  
- **Length**: 7 pages

# 1. Motivation
The memory bandwidth bottleneck has quite a huge adverse effect on an autoregressive decoding process like GPT rather than the encoder like BERT.
- In the case of BERT classification model: we can run the forward process only one time to get classification result in CLS token.
- In the case of GPT autoregressive model: we should run the forward pass several times predicting the next token in an autoregressive way to get the whole sentence
	- The number of runs is more than 1, which means that we should read the new key/values with new predicted token from HBM to L2 cache several times
So, reducing the inference time benefits autoregressive models like GPT more than it benefits other models like BERT.

In previous research, there was an attempt to reduce computation time such as Multi Query Attention (MQA)
- MQA uses multiple heads of Query with only one head of Key and Value.
It reduced the inference time a lot compared to the original Transformer Multi Head Attention(MHA), but also downgraded the performance.

# 2. Explanation
## 2-1. Goals
There are two goals in the research actually.
1. Uptraining the MHA based model into the MQA based model keeping the performance
2. Proposing GQA methodology which is the interpolation between MQA and MHA
But, in nanochat, andrej didn't use the first method, but used GQA methodology. So I'll not mention the first one in this post.

The goal of this Grouped Query Attention is the following:
1. Keep the similar or almost same performance compared to original Multi Head Attention that uses the same number of Q/K/V.
2. At the same time, by reducing the K/V, it reduces the computation resource(the degradation of speed of MHA caused by the memory bandwidth limitation). Therefore, this speeds up the inference and training time to a similar speed as MQA.

GQA is the interpolation between MHA(slow inference speed, high quality result) and MQA(high inference speed, low quality result)
- MHA: One Key/Value head per One Query
- MQA: One Key/Value head per Every Query
- GQA: One Key/Value head per Grouped-Queries

## 2-2. Notation:
GQA`-G` refers to grouped-query with G groups 
- If the G=1, GQA`-1`, it refers to one group of queries. It's the same as MQA 
- If the G=H, GQA`-H`, it refers to H groups of queries which have the same number as K/V. It's the same as MHA

# 3. GQA Methodology
## 3-1. Concept
Let's suppose the GQA`-(H/2)`, one K/V per two Queries.
Then the mapping relation is
$Q_1, Q_2 \rightarrow (K_1, V_1)$
$Q_3, Q_4 \rightarrow (K_2, V_2)$
$...$

Then the attention calculation is
$V_1 * \text{softmax}(\frac{Q_1@K_1^T}{\sqrt{d_k}}, \text{row-wise})$
$V_1 * \text{softmax}(\frac{Q_2@K_1^T}{\sqrt{d_k}}, \text{row-wise})$
$V_2 * \text{softmax}(\frac{Q_3@K_2^T}{\sqrt{d_k}}, \text{row-wise})$
$V_2 * \text{softmax}(\frac{Q_4@K_2^T}{\sqrt{d_k}}, \text{row-wise})$
...

Compared to MHA, just the mapping between Q and K/V is different, one K/V per n-Queries

## 3-2. Implementation trial
I tried to implement it, but failed. ChatGPT said it's not possible to implement with pure Pytorch.

I'll show you the code snippet that I tried first
```
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    num_of_expand = self.n_head//self.n_kv_head

    k = k.view(B,T,self.n_kv_head, 1, self.head_dim).expand(B,T,self.n_kv_head, num_of_expand,self.head_dim).view(B, T, self.n_head, self.head_dim)
    v = v.view(B,T,self.n_kv_head, 1, self.head_dim).expand(B,T,self.n_kv_head, num_of_expand,self.head_dim).view(B, T, self.n_head, self.head_dim)
```

The purpose of GQA is **Reducing memory bandwidth overhead by reducing K/V head**
So as you can see above code, the k and v have smaller number of heads than q.

But to calculate every head at once with matmul, we have to match the shape of Q and K/V anyway.

`.repeat()` physically copies the data into a new memory address, whereas `.expand()` produces the same result without copying — it just creates a view with stride 0. So even though the two produce identical values, `.expand()` avoids the extra HBM→L2 cache read that `.repeat()` causes.
Also we can't do reshape which uses `.contiguous()` that rearranges the memory address of data(creating new data), that also reads additional data from HBM to L2 cache.

So I selected `.expand()` function that match the shape of q and k/v but not copy or re-arrange the memory address.
But after I expand the k/v, because of the stride:0 problem, I couldn't use `.view()` function. But to solve the `.view()` problem, I can't use `.contiguous()` either, because there's an issue that's mentioned above.

So I stopped at that code snippet.
