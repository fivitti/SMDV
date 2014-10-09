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

SMDV further includes tools to help take the measurements for different data:

1. Convert the matrix to the supported formats
    * You can use this to see how the matrices are processed
      before performing the multiplications
    * Handled by the module _matrixformat.py_.
    * Command line interfaces in module _cli-conv.py_.
2. Return result multiplication
    * Methods for multiplying addition to the time the calculation return also the result of these calculations. You can use it to check its accuracy, either as an end in itself. It's a great, quick way to obtain a result of matrix by vector multiplication.
    * Handled by the module _matrixmultiplication.py_.
    * Command line interfaces in module _cli-multiply.py_.
3. Get CUDA kernels for multiplication
    * You can view the source files to see exactly how the multiplication is performed. You can also use the kernels implemented in their programs. In order to further optimize thecodes are written in CUDA C using metaprogramming.
    * Files in CUDA C are located in the _kernels_.
    * Module for metaprogramming and code compilation is handled by _cudacodes.py_.
4. Get info about matrix and vector files
    * You can quickly get information about the many of matrices stored in the format .mtx and Numpy vectors in .npy and export this information to a CSV file.
    * Command line interfaces in module _cli-info.py_.
5. Simple and fast generating matrix and vectors
    * You can generate random matrices and vectors for testing on selected sizes and densities.
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
## Authors
## License
## References  

# How to run
-------------
How to make SMDV to action and self test the speed of matrix multiplication?  
There are two ways. You can write a script in Python, or use the attached CLI.
> We assume that the vectors of the files are in the directory _data/vectors_ and matrices in _data/matrices_.  
> If not, change the appropriate path.

> We assume that you have a test matrix _matrix.mtx_ and vector _vector.npy_ of length equal to the number of columns of the matrix. If not, you can visit [MatrixMarket](http://math.nist.gov/MatrixMarket/) . We also recommend [matrix choice of Francisco Vazquez](http://www.hpca.ual.es/~fvazquez/?page=ELLRT) . You can also use tools provided to generate data described in next section or use package Numpy or Scipy.

## 1. Multiplication in script
We will use module _matrixmultiplication.py_. It is the core of the whole process. This module include moduls _cudacode.py_ and _matrixformat.py_. You do not have to worry about it. You only need to use core-module.

### 1.1 Full script
The complete script multiplication script is in _examples/multiplication.py_ will work according to the scheme:

1. Read data
2. Determination of parameters
3. Matrix multiplication
4. Results

### 1.2 ELLPACK multiplication
Call multiply this format is very simple. As any of the methods has in arguments matrix, vector and the number of repetitions of this operation. In addition, a takes one special parameter - **block size**.

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
This method has specials parameters: **threads per row**, **block size**, **prefetch** - number of requests for access to data notified in advance.

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
This method using fuction dot from numpy. It is here for comparison. It works only on CPU.

```python

    m = matrixmultiplication.multiply_cpu(matrix,
                                          vector,
                                          repeat=10)
    print 'List of execution times: {0}'.format(m[1])
```

## 2 Multiplication in command line interface (CLI)
