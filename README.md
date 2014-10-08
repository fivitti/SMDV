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
    * Handled by the module **matrixformat.py**.
    * Command line interfaces in module **cli-conv.py**.
2. Return result multiplication
    * Methods for multiplying addition to the time the calculation return also the result of these calculations. You can use it to check its accuracy, either as an end in itself. It's a great, quick way to obtain a result of matrix by vector multiplication.
    * Handled by the module **matrixmultiplication.py**.
    * Command line interfaces in module **cli-multiply.py**.
3. Get CUDA kernels for multiplication
    * You can view the source files to see exactly how the multiplication is performed. You can also use the kernels implemented in their programs. In order to further optimize thecodes are written in CUDA C using metaprogramming.
    * Files in CUDA C are located in the **kernels**.
    * Module for metaprogramming and code compilation is handled by **cudacodes.py**.
4. Get info about matrix and vector files
    * You can quickly get information about the many of matrices stored in the format .mtx and Numpy vectors in .npy and export this information to a CSV file.
    * Command line interfaces in module **cli-info.py**.
5. Simple and fast generating matrix and vectors
    * You can generate random matrices and vectors for testing on selected sizes and densities.
    * Handled by the module **matrixUtilites.py**.
    * Command line interfaces in module **cli-gen.py**.
6. And more...

## Dependencies
SMDV uses the packages:

* [PyCUDA](http://mathema.tician.de/software/pycuda/)
* [Numpy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [Click (for CLI)](http://click.pocoo.org/)
* and standard library, of course

## Project structure
## How to run
### Examples
## Results
## Authors
## License
## References  