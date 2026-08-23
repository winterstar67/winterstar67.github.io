---
title: "Distributed training environments"
date: 2026-06-10
draft: false
math: false
tags: ["PyTorch", "Distributed training"]
categories: ["Fundamentals"]
description: ""
---

# 1. Concepts to know
## 1-1. `RANK`
`RANK`: Global process ID among all distributed processes
Each of the GPUs has their own `RANK` value. With this `RANK` value, even if we run the same code, we can run code with different data by utilizing RANK.
Every code and loaded data are the same across all GPUs, but the `RANK` value is different.
So in nanochat, parameter split is conducted by the `RANK` like the code below
```
	 ...
     def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group['params']:
            if p.grad is not None:
                p_info = info["param_infos"][p]
                p_info['future'].wait()
                grad_slice = p_info['grad_slice']
                
                if p_info['is_small']:
                    p_slice = p
                else:
                    rank_size = p.shape[0]//world_size
                    p_slice = p[rank*rank_size: (rank+1)*rank_size]
     ...
```

### 1-1-1. How to import `RANK` in Python
```
import os

RANK = int(os.environ['RANK'])

# OUTPUT example
# RANK==0 in GPU 0 process
# RANK==1 in GPU 1 process
```

## 1-2. `LOCAL_RANK`
`LOCAL_RANK`: Process ID within the current machine/node

### 1-2-1. When there are multiple computers
The PID and LOCAL_RANK could be the same. But RANK is not duplicated across machines and processes

### 1-2-2. How to import `LOCAL_RANK` in Python
```
import os

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
```

## 1-3. `WORLD_SIZE`
`WORLD_SIZE`: Total number of distributed processes
- This is used to get the total number of GPUs in nanochat
```
# Example code in nanochat
# `N = world_size` is used to calculate the chunk_size in each GPU
    ...
    def _reduce_muon(self, group, world_size):

      p0 = group['params'][0]
      dtype, device = p0.dtype, p0.device

      grad_list = [p.grad for p in group['params'] if p.grad is not None]
      real_grads = torch.stack(grad_list,dim=0)
      real_grads_shape = real_grads.shape

      # Because stack is a copy of memory, I need to write this back onto the original parameter.
      K = real_grads.size(0)
      chunk_size = (K+N-1)//N
    ...
```

### 1-3-1. How to import `WORLD_SIZE` in Python
```
import os

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
```


# 2. Note
## 2-1. PID and RANK are different
OS PID is not the same as RANK in PyTorch's distributed training.
But for one process, the unique OS PID and RANK number are allocated
