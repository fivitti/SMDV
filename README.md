# SMDV
------
## Description
SMDV is a collection of modules to test the speed of execution of
multiplication for different matrices formats.  
Multiplications are performed on the GPU using technology NVIDIA CUDA.
### Supported formats
Implemented multiplication formats (in CUDA C):  

* ELLPACK
* SLICED ELLPACK
* ERTILP
* SERTILP
* CSR

SMDV also allows use of the multiplication module Numpy.
Such calculations will be performed only on the CPU.
You can use them as a reference point for measurements performed on the GPU.

### Functions
The main purpose of the project is to measure the execution time
of multiplications on the GPU for different matrix formats.
You can repeat our research on your own computer (you only need to have a 
compatible Nvidia GPU with CUDA).  
You can also carry out their own experiments and their own data 
independently chosen parameters.  
SMDV allows easy collaboration with external scripts.

SMDV further includes tools to help take the measurements for different data:

1. Convert the matrix to the supported formats
    * You can use this to see how the matrices are processed
      before performing the multiplications
    * Handled by the module _matrixformat.py_.
    * Command line interfaces in module _cli-conv.py_.
2. Return result multiplication
    * Methods for multiplying addition to the time the calculation return also 
    the result of these calculations. You can use it to check its accuracy, 
    either as an end in itself. It's a great, quick way to obtain a result of 
    matrix by vector multiplication.
    * Handled by the module _matrixmultiplication.py_.
    * Command line interfaces in module _cli-multiply.py_.
3. Get CUDA kernels for multiplication
    * You can view the source files to see exactly how the multiplication is 
    performed. You can also use the kernels implemented in their programs. 
    In order to further optimize thecodes are written in CUDA C using 
    metaprogramming.
    * Files in CUDA C are located in the _kernels_.
    * Module for metaprogramming and code compilation is handled 
    by _cudacodes.py_.
4. Get info about matrix and vector files
    * You can quickly get information about the many of matrices stored in 
    the format .mtx and Numpy vectors in .npy and export this information 
    to a CSV file.
    * Command line interfaces in module _cli-info.py_.
5. Simple and fast generating matrix and vectors
    * You can generate random matrices and vectors for testing on selected 
    sizes and densities.
    * Handled by the module _matrixUtilites.py_.
    * Command line interfaces in module _cli-gen.py_.
6. And more...

## Dependencies
SMDV uses the packages:

* [PyCUDA](http://mathema.tician.de/software/pycuda/)
* [Numpy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [Click (for CLI)](http://click.pocoo.org/)
* and standard library, of course

## Results
The following are examples of the results of test using the 10 largest 
matrices from [matrix choice of Francisco Vazquez](http://www.hpca.ual.es/~fvazquez/?page=ELLRT).

The calculations were performed on a computer TO BE COMPLETED <<< >>

Version software:
    
* OS: Ubuntu 13.10
* python 2.7.5+
* PyCuda 2013.1.1
* Numpy 1.7.1
    
Arguments:
    
* Block size: 128
* Threads per row 2: 2
* Slice size: 64
* Prefetch: 2
* Align: 32
* Repeats: 10
* Compensation: 10 - So much has been made multiplications
  "starting". Their results were not included.

Test vectors are generated at random in the range of -1 to 1.
Their density is 15%. You can download them from the folder _vectors_.
    
Informations about matrices:

| Matrix | Rows  | Columns | NNZ     | Sparsity | 
| ------ | ----  | ------- | ------- | -------- |
| cant   | 62451 | 62451   | 4007383 | 0.103%   |
| consph | 83334 | 83334   | 6010480 | 0.087%   |
| cop20k_A | 121192 | 121192 | 2624331 | 0.018% |
| mac_econ_fwd500 | 206500 | 206500 | 1273389 | 0.003% |
| mc2depi | 525825 | 525825 | 2100225 | 0.001%  |
| pwtk   | 217918 | 217918 | 11634424 | 0.024%  |
| qcd5_4 | 49152 | 49152   | 1916928 | 0.079%   |
| rma10  | 46835 | 46835   | 2374001 | 0.108%   |
| shipsec1 | 140874 | 140874 | 7813404 | 0.039% |
| wbp256 | 65537 | 65537   | 31413934 | 0.731%  |

Results:

| Matrix | Format | Avr time [ms] | Std [ms] |
| ------ | ------ | ------------- | --- |
| cant   | CPU    |  7.658        | 0.021 |
|        | CSR    | 0.444         | 0.002 |
|        | ELLPACK | 0.348        | 0.001 |
|        | SLICED | 0.290         | 0.003 |
|        | SERTILP | 0.275        | 0.002 |
|        | ERTILP  |  0.320       | 0.001 |
| consph | CPU    |  11.493       | 0.014 |
|        | CSR    | 0.521         | 0.001 |
|        | ELLPACK | 0.419        | 0.002 |
|        | SLICED | 0.373         | 0.001 |
|        | SERTILP | 0.356        | 0.001 |
|        | ERTILP  |  0.423       | 0.001 |
| cop20k_A | CPU    |  6.627        | 0.010 |
|        | CSR    | 0.710         | 0.002 |
|        | ELLPACK | 0.545        | 0.001 |
|        | SLICED | 0.537         | 0.002 |
|        | SERTILP | 0.523        | 0.012 |
|        | ERTILP  |  0.528       | 0.004 |
| mac_econ_fwd500 | CPU    |  2.734        | 0.008 |
|        | CSR    | 1.025         | 0.025 |
|        | ELLPACK | 0.696        | 0.001 |
|        | SLICED | 0.674         | 0.002 |
|        | SERTILP | 0.629        | 0.002 |
|        | ERTILP  |  0.648       | 0.002 |
| mc2depi   | CPU    |  4.556        | 0.160 |
|        | CSR    | 1.868         | 0.003 |
|        | ELLPACK | 0.818        | 0.002 |
|        | SLICED | 0.844         | 0.001 |
|        | SERTILP | 0.836        | 0.003 |
|        | ERTILP  |  0.827       | 0.001 |
| pwtk   | CPU    |  22.317        | 0.009 |
|        | CSR    | 1.268         | 0.001 |
|        | ELLPACK | 1.073        | 0.003 |
|        | SLICED | 0.904         | 0.002 |
|        | SERTILP | 0.872        | 0.001 |
|        | ERTILP  |  1.006       | 0.002 |
| qcd5_4   | CPU    |  3.224        | 0.020 |
|        | CSR    | 0.295         | 0.001 |
|        | ELLPACK | 0.190        | 0.001 |
|        | SLICED | 0.190         | 0.002 |
|        | SERTILP | 0.185        | 0.000 |
|        | ERTILP  |  0.183       | 0.001 |
| rma10   | CPU    |  3.925        | 0.007 |
|        | CSR    | 0.330         | 0.001 |
|        | ELLPACK | 0.283        | 0.001 |
|        | SLICED | 0.245         | 0.004 |
|        | SERTILP | 0.226        | 0.001 |
|        | ERTILP  |  0.258       | 0.002 |
| shipsec1   | CPU    |  15.051        | 0.047 |
|        | CSR    | 0.932         | 0.003 |
|        | ELLPACK | 0.787        | 0.005 |
|        | SLICED | 0.712         | 0.001 |
|        | SERTILP | 0.681        | 0.000 |
|        | ERTILP  |  0.773       | 0.003 |
| wbp256   | CPU    |  81.732       | 0.026 |
|        | CSR    | 1.873         | 0.005 |
|        | ELLPACK | 1.608        | 0.002 |
|        | SLICED | 1.153         | 0.003 |
|        | SERTILP | 1.085        | 0.002 |
|        | ERTILP  |  1.515       | 0.007 |


    
## Authors
* Krzysztof Sopyła - [ksirg](https://github.com/ksirg) on Github
* Paweł Drozda - [pdrozda](https://github.com/pdrozda) on Github
* Sławomir Figiel - [fivitti](https://github.com/fivitti) on Github

## License
SMDV is licensed under the MIT License.

## References
*  "Efficient Sparse Matrix-Vector Multiplication on CUDA", Nathan Bell, 
Michael Garland [11 December 2008]
* "The sparse matrix vector product on GPUs", F. Vazquez, E. M. Garzon, 
J. A. Martınez, J. J. Fernandez [14 June 2009]
* "Improving the performance of the sparse matrix vector product with GPUs",
F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon [2010]
* "Automatically Tuning Sparse Matrix-Vector Multiplication for GPU 
Architectures", Alexander Monakov, Anton Lokhmotov, and Arutyun Avetisyan
* "A memory efficient and fast sparse matrix vector product on a gpu",
A. Dziekonski, A. Lamecki, and M. Mrozowski

# How to run
-------------
How to make SMDV to action and self test the speed of matrix multiplication?  
There are two ways. You can write a script in Python, or use the attached CLI.

> We assume that you have a test matrices and vectors of length equal to 
the number of columns of the matrix. If not, you can 
visit [MatrixMarket](http://math.nist.gov/MatrixMarket/) . We also 
recommend [matrix choice of Francisco Vazquez](http://www.hpca.ual.es/~fvazquez/?page=ELLRT) . 
You can also use tools provided to generate data described in next section or 
use package Numpy or Scipy.

## 1. Multiplication in script
We will use module _matrixmultiplication.py_. It is the core of the whole 
process. This module include moduls _cudacode.py_ and _matrixformat.py_. 
You do not have to worry about it. You only need to use core-module.

### 1.1 Full script
The complete script multiplication script is in _examples/multiplication.py_ 
will work according to the scheme:

1. Read data
2. Determination of parameters
3. Matrix multiplication
4. Results

### 1.2 ELLPACK multiplication
Call multiply this format is very simple. As any of the methods has in 
arguments matrix, vector and the number of repetitions of this operation. 
In addition, a takes one special parameter - **block size**.

```python

    m = matrixmultiplication.multiply_ellpack(matrix,
                                              vector,
                                              block_size=128,
                                              repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

### 1.3 SLICED multiplication
This method has three specials parameters: **threads per row**, **slice size**,
**align** - constant to calculation and set align of the matrix.

```python

    m = matrixmultiplication.multiply_sliced(matrix,
                                             vector,
                                             align=64,
                                             slice_size=32,
                                             threads_per_row=2,
                                             repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

### 1.4 SERTILP multiplication
This method has specials parameters: **threads per row**, **slice size**,
**align**, **prefetch** - number of requests for access to data notified in advance.

```python

    m = matrixmultiplication.multiply_sertilp(matrix,
                                              threads_per_row=2,
                                              slice_size=32,
                                              prefetch=2,
                                              align=64,
                                              repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

### 1.5 ERTILP multiplication
This method has specials parameters: **threads per row**, **block size**, 
**prefetch** - number of requests for access to data notified in advance.

```python

    m = matrixmultiplication.multiply_ertilp(matrix,
                                             threads_per_row=2,
                                             prefetch=2,
                                             block_size=128,
                                             repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

### 1.6 CSR multiplication
This method has only one specials parameters: **block size**.

```python

    m = matrixmultiplication.multiply_csr(matrix,
                                          vector,
                                          block_size=128,
                                          repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

### 1.6 CPU (Numpy) multiplication
This method using fuction dot from numpy. It is here for comparison. 
It works only on CPU.

```python

    m = matrixmultiplication.multiply_cpu(matrix,
                                          vector,
                                          repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

## 2 Multiplication in command line interface (CLI)
CLI to multiply is in module _cli-multiply.py_. Description all parameters 
get using flag **--help**.

As arguments required program takes two paths. First indicates vector, and 
the second matrix. 

If the directory path vectors will then be searched for files _.npy_. 
The program alone try to match the vector for the given matrix. 
Will match the last alphabetically vector of length equal to the number of 
columns of the matrix If you do not succeed will be displayed message.

If the path is a directory matrix program will search it for files _.mtx_. 
Then try to perform calculations for each array in turn.

Specifying of two paths as directory  is a simple way to automate testing 
on a collection.

You can easily specify all the parameters.  
Multiplications are making a choice of formats by using the flag: 
**-ell** (ELLPACK), **-sle** (SLICED), **-ert** (ERTILP), **-see** (SERTILP), 
**-csr** (CSR) and **-cpu** (multiplication on CPU).

CLI allows you to display the result of multiplication, all execution times, 
average execution time and standard deviation of the mean.

You can also test the correctness of the returned results with a confidence 
factor. To do this, use the flag **--test**. The result will be the result 
of a standard function **dot** with Numpy package. If errors do not coming 
within specified range will be displayed corresponding messages.

You can also save the results of the measurements in the CSV file with 
the flags **-o**.
### 2.1 Examples
Multiplications matrix by vectors in ELLPACK, ERTILP and CPU formats with own 
parameters and print average time and standard deviation.

```
    
    user@host:~/projekty/SMDV$ python cli-multiply.py -b 128 -tpr 4 -p 2 -ell 
    -ert -cpu -avr -std Data/vectors/Vector_62451.npy Data/real/cant.mtx
    Multiply matrix Data/real/cant.mtx by the vector Data/vectors/Vector_62451.npy
    Multiplication with Numpy (only CPU)
    Average time [ms]: 7.84460783005
    Standard deviation: 0.0
    Multiplication with ELLPACK
    Average time [ms]: 0.6616320014
    Standard deviation: 0.0
    Multiplication with ERTILP
    Average time [ms]: 0.466080009937
    Standard deviation: 0.0

```

Auto search vector:

```

    user@host:~/projekty/SMDV$ python cli-multiply.py Data/vectors/ Data/real/cant.mtx
    Multiply matrix Data/real/cant.mtx by the vector Data/vectors/Vector_62451.npy
    
```

Auto search matrices and vectors:

```

    user@host:~/projekty/SMDV$ python cli-multiply.py Data/vectors/ Data/real/
    Multiply matrix Data/real/mac_econ_fwd500.mtx by the vector Data/vectors/Vector_206500.npy
    Multiply matrix Data/real/cop20k_A.mtx by the vector Data/vectors/Vector_121192.npy
    Multiply matrix Data/real/wbp256.mtx by the vector Data/vectors/Vector_65537.npy

```

Under construction:

    3. Data generation
    4. Matrix convertion
    5. Information about data
    6. Building kernels