# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 20:33:37 2014

@author: Sławomir Figiel
"""
import matrixmultiplication

import scipy.io
import numpy

# Matrix in Matrix Market format
matrix_file = 'data/matrices/matrix.mtx'
# Vector in Numpy Binary File format
vector_file = 'data/vectors/vector.npy'

#####################
##### Read data #####
#####################
matrix = scipy.io.mmread(matrix_file)
vector = numpy.load(vector_file)
# We can use numpy.arange for generating vector too.

#######################################
##### Determination of parameters #####
#######################################
# Repeat multiplications. So many execution times will be returned
repeat = 10
# Size of block CUDA (on GPU). Recommended 128 or 256.
# Using in ELLPACK, CSR, ERTILP formats.
block_size = 128
# Constant to calculation and set align of the matrix. Recommended multiple 2.
# Using in SLICED, SERTILP formats.
align = 64
# Size of simple slice. Recommended multiple 2. Using in SLICED, SERTILP.
slice_size = 32
# Threads per row. Recommended 2, 4 or 8. Using in SLICED, SERTILP, ERTILP.
threads_per_row = 2
# Number of requests for access to data notified in advance.
# Recommended 2, 4 or 8. Using in SERTILP, ERTILP formats.
prefetch = 2

#################################
##### Matrix multiplication #####
#################################
# Multiplication ELLPACK
m_ellpack = matrixmultiplication.multiply_ellpack(matrix,
                                                  vector,
                                                  block_size=block_size,
                                                  repeat=repeat)
# Multiplication SLICED ELLAPACK
m_sliced = matrixmultiplication.multiply_sliced(matrix,
                                            vector,
                                            align=align,
                                            slice_size=slice_size,
                                            threads_per_row=threads_per_row,
                                            repeat=repeat)
# Multiplication SERTILP
m_sertilp = matrixmultiplication.multiply_sertilp(matrix,
                                            threads_per_row=threads_per_row,
                                            slice_size=slice_size,
                                            prefetch=prefetch,
                                            align=align,
                                            repeat=repeat)
# Multiplication ERTILP
m_ertilp = matrixmultiplication.multiply_ertilp(matrix,
                                            threads_per_row=threads_per_row,
                                            prefetch=prefetch,
                                            block_size=block_size,
                                            repeat=repeat)
# Multiplication CSR
m_csr = matrixmultiplication.multiply_csr(matrix,
                                          vector,
                                          block_size=block_size,
                                          repeat=repeat)
# Mutilpication on CPU, using fuction dot from numpy, for comparison.
m_cpu = matrixmultiplication.multiply_cpu(matrix,
                                          vector,
                                          repeat=repeat)
###################
##### Results #####
###################
# The result of multiplication is the first element of the returned tuples.
print "Result multiply using ELLPACK format: {0}".format(m_ellpack[0])
print "Result multiply using SLICED ELLPACK format: {0}".format(m_sliced[0])
# And so on...

# List of execution times is the second element of the tuple
print 'Execution times:'
print 'ELLPACK: {0}'.format(m_ellpack[1])
print 'SLICED ELLPACK: {0}'.format(m_sliced[1])
print 'SERTILP: {0}'.format(m_sertilp[1])
print 'ERTILP: {0}'.format(m_ertilp[1])
print 'CSR: {0}'.format(m_csr[1])
print 'CPU: {0}'.format(m_cpu[1])

# Average executions times
print 'Average execution times:'
print 'ELLPACK: {0}'.format(numpy.average(m_ellpack[1]))
print 'SLICED ELLPACK: {0}'.format(numpy.average(m_sliced[1]))
print 'SERTILP: {0}'.format(numpy.average(m_sertilp[1]))
print 'ERTILP: {0}'.format(numpy.average(m_ertilp[1]))
print 'CSR: {0}'.format(numpy.average(m_csr[1]))
print 'CPU: {0}'.format(numpy.average(m_cpu[1]))
