---
title: "C++(Compiler) in pytorch"
date: 2026-04-28
draft: false
math: false
tags: ["Background"]
categories: ["Fundamentals"]
description: ""
---

## The purpose of understanding Compiler:
- Getting deeper understand on the principle and flow of operations inside pytorch.
```
  x ** 2
    └─→ Calling Python __pow__
          └─→ torch.pow(x, 2)
                └─→ aten::pow(x, 2)   ← C++ kernel

  torch.pow(x, 2)
    └─→ aten::pow(x, 2)               ← C++ kernel

  x.square()
    └─→ aten::square(x)
          └─→ aten::pow(x, 2)         ← C++ kernel
```

Why did they choose to run with C++, not running with python?
- That's probably the reason of choosing compiler

Check whether you can explain how pytorch processes an operation.


# 0. Background
- Every code written in Python or C++ should eventually be converted into machine language because the written code is only human readable while CPU can't.
- There are ways to convert codes into CPU readable language and the two famous ways are compiler and interpreter.

`loop` and `add operation` are good examples to perceive the difference of compiler and interpreter. The below `Code 1` example has both `loop` and `add` operation

```
# Code 1

# think it as a pseudo code of python and C++.
int N = input() # Let's suppose N=10000
for (int i = 0; i < N; i++) { 
	sum += x[i]; 
}
```

## 0-1. Terminology
- Dispatch: The process that determines which implementation to use
	- For the `add`, there are several `add`s such as string add, int add, and list add. - Python needs this process. But C++ doesn't have this process because it knows what to run at first with compiling the code
- Executable file: The file that the machine directly runs, it's written in machine language
- GIL: It controls that only one Python thread can run Python bytecode at a time. There is one GIL per Python process


# 1. What is the compiler?
- Compiler is one of the necessary components to execute the code written with programming language in CPU.
- Compiler just prepares every code to be ready to run in CPU at once with optimization before CPU actually runs it.
- Once the code is compiled, there's no type check or other possibility of interpretation anymore. It's static, always shows same operation.
```
# Flow of C++ code execution

C++ source code
    ↓
Compiler (e.g., g++)
    ↓
Object file (.o)
    ↓
Linker
    ↓
Executable file ← A executable file that CPU can run
```

Result of Compiling `Code 1`
```
# Compiled file operation

compare i with N
if i >= N, exit  
load x[i]  
add to sum  
i++  
jump back
```

# 2. What is the interpreter?
- Interpreter is one of the ways to hand over CPU readable language to run operation
- But, it does not prepare every code to the CPU readable language, Literally, it just checks every line of code such as type check, which type's add operation to use, etc.
- It also requires each string, int to be in the object. So int32 in python should take more than 4bytes(32bit) for interpreter to know what type it is and what operation to use, string add? int add? list add?

Result of interpreting `Code 1` 
```
# Python interpreter for every loop N=10000 times

load object sum
load object x
load object i
check x type
check i type
bounds check
get list element
check sum type
check value type
perform int add
allocate new int object
rebind sum
```

# 3. Summary of `compiler` and `interpreter`
- Compiler: 
	- Pre-define and optimized executable file from written code.
	- It's fast to run once it's compiled.
	- Memory efficient. it just use promised bytes (4 bytes and 8 bytes for int32 and int64 each) (no overhead)
	- But the flexibility such as changing type things is lower than python, we should care each variable not to change the type
- interpreter: 
	- Not pre-defined and optimized executable file from written code
	- It's slow because it always checks every line even when it encountered same variable in the prev line
		- When we do add operation in the for loop, it have to check whether this add is for integer, string, or list, etc. Even in the for loop
	- It's memory inefficient, it takes more bytes than pure data type requires (overhead). To know what operation to use, what type it is in runtime(real time), it should have not only integer but also metadata such as what type it is. 
	- But it's flexible.

- If we use interpreter, we always have to check the object type of each variable that the interpreter encounter, when we do add things, we have to check whether this add is for integer, string, or list, etc. Even in the for loop too!
- But if we use compiler like C++ uses, for the 100000 loop, we don't check each variable's type things for every loop even for the same variable, Because for the every variables, the operations and the type of each variables are determined .


## 3-1. Comparison of Compiler and Interpreter with table

|                               | Compiler                                                                                                    | Interpreter                                                                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| memory overhead               | No (compiler already knows which operation to use for each variable, it's already determined so don't need) | Yes (For each line by line interpretation, we need to refer the object type(str, int, or float) to pick appropriate add, multiplication operation) |
| speed                         | Fast                                                                                                        | Slow                                                                                                                                               |
| flexibility                   | Low                                                                                                         | High                                                                                                                                               |
| When the operation determined | At compiling before execute the code                                                                        | At each runtime                                                                                                                                    |

# 4. So when, why, and how the pytorch use C++(compiler)?
## 4-1. When pytorch uses C++:
- There are bunch of pytorch functions and operations. So I can't cover every case. But basically the operation that C++ also can handle, that operation would be replaced with C++
- Pytorch operations: 
	1. `.square()`, 
	2. `torch.tensor()`
	3. `torch.sum()`
	4. `torch.matmul()`
	5. `torch.nn.Linear.forward()`
	6. etc

## 4-2. Why pytorch uses C++:
There are two reasons that pytorch is faster than pure python.
1. Usage of CUDA 
	- For the CUDA operation, refer to the {{< wikilink "cuda" >}}
2. Usage of C++ and GIL

We can consider two cases of why pytorch uses C++ 
- Case 1: When pytorch doesn't use CUDA
	- C++ compiler is faster than python interpreter.
- Case 2: When pytorch uses CUDA
	- C++ isn't restricted by GIL. Python can only run one single thread
	- C++ compiler is faster than python interpreter.


- Pytorch calls the pre-compiled and saved C++ function files and runs that pre-saved C++ files.
	- Check `5. Flow of pytorch execution` for comparison of python and C++

## 4-3. How pytorch uses C++:
- pybind11 library makes python to call C++ functions.
- Pytorch calls C++ with pybind11

## 4-4 (Q). Is C++ executable file created whenever we run pytorch?
No, running C++ code requires the executable file but, the pytorch does not generate the executable file every time you run pytorch code, there's just pre-saved executable file in the pytorch library.
Unlike the custom C++ , the Executable file is already installed in the pytorch library (The `libtorch.dll` file has C++ compiled files), the Executable file for the `nn.Linear` or `tensor.square()` is already defined and saved as an file

# 5. Execution flow of pytorch operation
Flow: 
1. Python: User write pytorch code in python - interface to call C++ function
2. C++: Python call C++ - Real executor of operation, loop, and kernel selection, etc.
3. CUDA: Operator when we move the data into 'cuda' (Optional)

Comparison of python and C++ in pytorch
- Every part before calling the CUDA. So track the flow calling of CUDA from python code and check the intermediate codes to compare python and C++
- Basically, the CUDA operation would happen only for the variables `x` which is `x.to("cuda")`

**Let's see some of examples of pytorch execution**.
One assumption on comparing C++ and python in pytorch:
1. If we assume to use C++(pytorch does in real), we must fix to use C++ compiler and C++ data(no int object, just int32 4bytes) and there's no room to use python except the pybind11. Every internal operation should be supposed to use compiler.

2. If we assume to use python(pytorch doesn't in real), we must fix to use python interpreter and python data type(not pure int32 4bytes, there's memory overhead) and there's no room to use C++. Every internal operation should be supposed to use interpreter.

## 5-1. torch.tensor generation
It requires object(class), but if we use C++ with pre-determined type, then we don't have to.
... I think we always have to use python list in `torch.tensor generation`
- The usage of C++ would happened inside the pytorch, not outside of pytorch

The background mind
- When we use python(interpreter), the values are python object(memory overhead)

```
# Python case, pytorch doesn't use

# ----- #

1. list = []
# (1) create Python list object 
# (2) allocate pointer array with length 5

2. loop for list.append()
# (1) create/load PyLongObject(1) 
# (1) store pointer to PyLongObject(1) in list[1]
# (2) create/load PyLongObject(2) 
# (2) store pointer to PyLongObject(2) in list[2]
# ...
# (1000) create/load PyLongObject(1000)
# (1000) store pointer to PyLongObject(1000) in list[1000]

3. torch.tensor(list) # torch.tensor conversion
# C++ reads int value from each python int object
# allocate memory to pytorch raw int list
# store pure int32 into pytorch list


# Speed of list generation: Little bit slower than compiler 
# Memory overhead: python requires more memory because it's not pure int32 but python int object(memory overhead)
# The list store pointers to the object
```

```
# C++ case, pytorch uses

# ----- #

1. list = [] # allocate 4*{element_size} bytes contiguous memory

2. loop for list append
# store 1 at arr[0]  
# store 2 at arr[1]  
# ...
# store 1000 at arr[1000]

3. torch.tensor()
# No python object reads and extracts. 

# Speed of list generation: Little bit faster than interpreter
# Memory overhead: No overhead, just store the pure values
# The list store the pure int value itself.
```

## 5-2. `tensor.square()`
For the square of x, it would be `x*x` operation
```
# Python case, pytorch doesn't use

# ----- #
load x
check type of x
perform multiplication
save result on x
```

```
# C++ case, pytorch uses

# ----- #
multiplication of x
save result on x
```


## 5-3. `torch.sum(tensor)`
Summation is a combination of `loop` of `add` 
```
# Python case, pytorch doesn't use

# ----- #
# Python interpreter for every loop N=10000 times

load object sum
load object x
load object i
check x type
check i type
bounds check
get list element
check sum type
check value type
perform int add
allocate new int object
rebind sum
```

```
# C++ case, pytorch uses

# ----- #
compare i with N
if i >= N, exit  
load x[i]  
add to sum  
i++  
jump back
```

## 5-4. If we use CUDA, does it matter what to use between python or C++?
There are many processes to check before running CUDA
```
# Python case that pytorch doesn't use

# ----- #
Probably the python(interpreter) need metadat with object for line by line interpretation. Then the torch tensor should be list of pointer, and each pointer must point the object(python int object) not pure data type(int32).
From that, there would be memory overhead, extract each of pure data from object, handing over, and doing this steps with loop.
The C++ compiler has every data and few of metadata that covers every data with few of nums such as shape, dtype.
```

```
# C++ case that pytorch uses

# ----- #
- Tensor metadata access
	- dtype: fp32 / fp16 / bf16 / int8  
	- shape: M, N, K  
	- stride/layout: contiguous? transposed?  
	- device: cuda  
	- autograd: save tensors for backward?
- dispatch key selection: CUDA / CPU / AutogradCUDA / SparseCUDA / etc.  
- output tensor allocation  
- CUDA stream selection  
- memory allocator interaction  
- shape/stride handling  
- broadcasting logic  
- error checking  
- calling CUDA runtime / cuBLAS / cuDNN / custom kernels  
- autograd graph construction
```

# 6. Check point

## 6-1. Some of code edit doesn't require re-compiling 
The executable file can be reused if the only N is changed in `for N loop`, not creating new executable file with re compiling the C++ codes.
- If we compiled the code `loop (N=)10000 times`, it create the executable file for that code. Now suppose that we only changed `N=10000` into `N=33` and compiled it again. Whether a compiler re-compile it again or not is up to the code structure.
	- If you fixed the `N=10000` on your code, you should re-compile it creating new executable file changing it into `N=33`. N is the part of the code.
	- If you write `N=input()`, you don't have to re-compile it. Just run the existing executable file, handing over input N. N is just metadata of runtime. - pytorch uses this method
		- Changing N is not critical to the speed of executing codes

## 6-2. Compiler is not always faster than interpreter
- Compiler has an extra overhead that the interpreter doesn't, especially in pytorch. If the data size is small, then the dispatch in pytorch such as which type is it? which kernel should I use? things can be overhead and slower than interpreter(python operation)


## 6-3. What is the Global Interpreter Lock(GIL)?
- This is one of the reasons that we use C++ in CUDA kernel selection
```
   Python interpreter:
    [GIL: Thread 1 has the GIL currently]

    Thread 1: Python is running the code  ← Holding the GIL
    Thread 2: Waiting          ← Didn't obtain the GIL
    
    When C++ code is called, C++ releases the GIL through the Python C API
    
  // Inside PyTorch C++ code
  Py_BEGIN_ALLOW_THREADS   // The GIL is released
    // CUDA kernel execution, heavy operation, etc.
  Py_END_ALLOW_THREADS     // The GIL is obtained again

  Even though it is in the same process, C++ is saying "I'll not touch any Python object, you can release the GIL" to the interpreter

  Python (Holding the GIL)
    → Calling C++
        → Py_BEGIN_ALLOW_THREADS  ← The GIL is released
        → CUDA kernel execution
        → Py_END_ALLOW_THREADS    ← The GIL is obtained again
    → return to Python
```

# 7. Key points

## 7-1. The important effect that distinguish C++(Interpreter) and Python(Interpreter)
- Code execution speed (C++ is faster - Python has overhead like type check)
- Memory overhead (Python has memory overhead using object class for type info store)
- Flexibility: Whether we can change the variable values without restriction

## 7-2. Pytorch uses C++
