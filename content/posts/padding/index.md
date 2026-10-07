---
title: "Padding"
date: 2026-06-10
draft: false
math: false
tags: ["Techniques", "Tips"]
categories: ["Fundamentals"]
description: ""
---

# How to do 0-padding?
Suppose we have `A = torch.tensor([...])`. In this case, if we'd like to add the padding on the tensor `A`, we can think of two options.
1. Padding additional tensors on the existing tensor using `F.pad`
2. Pre-allocate the whole size of the tensor which includes the padded tensor at first, then copy real data in. The right-tail part in the tensor stays zero automatically (or is explicitly zeroed). Simpler than collecting, measuring, and padding one by one.

# The code in nanochat
`stacked_grads = torch.zeros(chunk_size*N, *real_grads_shape[1:], dtype=dtype, device=device)` is the code that pre-allocates the padded buffer. 
```
    def _reduce_muon(self, group, world_size):

        p0 = group['params'][0]
        dtype, device = p0.dtype, p0.device

        grad_list = [p.grad for p in group['params'] if p.grad is not None]
        real_grads = torch.stack(grad_list,dim=0)
        real_grads_shape = real_grads.shape
        
        N = world_size
        K = real_grads.size(0)

        chunk_size = (K+N-1)//N
        stacked_grads = torch.zeros(chunk_size*N, *real_grads_shape[1:], dtype=dtype, device=device)
        stacked_grads[:real_grads_shape[0]] = real_grads

        grad_chunk = torch.empty_like(real_grads[:chunk_size])
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return {"future":future, "grad_chunk":grad_chunk, "stacked_grads":stacked_grads, "chunk_size":chunk_size}
```
