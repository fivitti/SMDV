# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 20:16:06 2014
@author: Sławomir Figiel

Module provides the methods for converting matrix to formats:
    * CSR (Scipy)
    * Ellpack
    * Sliced Ellpack
    * SERTILP
    * ERTILP
There are also specific, support method for the conversion.
"""
import listutilites as lu
import scipy.sparse
import numpy
from math import ceil

def reshape_ell_to_multiple(matrix_to_extension, multiple_row):
    '''
    Method extends matrix preconvert Ellpack. Adds empty list to the
    list of rows and columns and zero to the list length of rows.
    Adds many elements that the total number in each list
    (rows, columns, length of lines) be a multiple parameter multiple_row.

    Parameters
    ==========
    matrix_to_extension : list of list
        Represents matrix preconvert Ellpack. Should have three parts:
        list of values​​, list of columns and list of row length.
        List of values and columns are list of list with rows. List of
        row length is list of integers.
    multiple_row : integer > 0
        The length of each list the resulting matrix is a multiple of
        this number.

    Returns
    =======
    Nothing. Method works in place.

    Notes
    =====
    If the length of the lists are already multiple multiple_row method
    does not change anything.

    Examples
    ========
    >>> pre_ell = [[[1], [3, 1], [7]], [[1], [0, 2], [2]], [1, 2, 1]]
    >>> reshape_ell_to_multiple(pre_ell, 4)
    >>> pre_ell
        [[[1], [3, 1], [7], []], [[1], [0, 2], [2], []], [1, 2, 1, 0]]
    '''
    rows = len(matrix_to_extension[2])
    modulo = rows % multiple_row
    if modulo == 0:
        return
    to_add = multiple_row - modulo
    matrix_to_extension[0].extend([[]]*to_add)
    matrix_to_extension[1].extend([[]]*to_add)
    matrix_to_extension[2].extend([0,]*to_add)

def set_align(matrix_sliced_ell, align=64):
    '''
    Sets align for matix Sertilp or Sliced. Increases the slices by
    adding zeroes to the end, so that their size is not less than
    the align parameter.
    The method works in a place.

    Parameters
    ==========
    matrix_sliced_ell : matrix-type sliced
        List of list: [values, columns, row_lenghts, slice_starts].
    align : integer > 0
        A single slice will be not less than that number.

    Returns
    =======
    Nothing. Method works in a place. Processed array slice will
    be not less than "align".

    Examples
    ========
    >>> vals = [1, 0, 2, 0, 3, 4, 0, 0]
    >>> cols = [0, 0, 1, 0, 0, 2, 0, 0]
    >>> row_lengths = [1, 1, 2, 0]
    >>> slice_starts = [0, 4, 8]
    >>> matrix = [vals, cols, row_lengths, slice_starts]
    >>> set_align(matrix, 8)
    >>> matrix[0]
        [1, 0, 2, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0]
    >>> matrix[1]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    >>> matrix[2]
        [1, 1, 2, 0]
    >>> matrix[3]
        [0, 8, 16]

    For threads per row = 2, slice size = 2,
    matrix = [[1, 0, 0], [0, 2, 0], [3, 0, 4]]
    '''
    counter = 0
#    end = matrix_sliced_ell[3][counter]
    rows = len(matrix_sliced_ell[3]) - 1
    shift = 0
    while rows > counter:
        counter += 1
        matrix_sliced_ell[3][counter] += shift
        start = matrix_sliced_ell[3][counter-1]
        end = matrix_sliced_ell[3][counter]
        different = align - (end - start)
        if different < 0:
            continue
        shift += different
        for _ in range(different):
            matrix_sliced_ell[0].insert(end, 0)
            matrix_sliced_ell[1].insert(end, 0)
            matrix_sliced_ell[3][counter] += 1

def preconvert_to_ell(matrix):
    '''
    Performs basic steps in the conversion matrix type ellpack and
    derivatives.
    Returns a tuple of lists containing the sequence: values of matrix
    without zeros, column indices corresponding to values​​, length of rows.

    Parameters
    ==========
    matrix : numpy array or scipy matrix
        Input matrix

    Returns
    =======
    preconverted matrix : tuple
        Tuple has three lists. First is list (of list) of row of matrix
        without zeros.
        Second is list of list of indices columns coresponding to values.
        Third is normal integer list of rows length.

    Examples
    ========
    >>> import numpy
    >>> matrix = numpy.array([[1, 0, 0, 0],
    ...                       [0, 2, 3, 0],
    ...                       [0, 0, 0, 0],
    ...                       [4, 0, 0, 5]])
    >>> preconvert_to_ell(matrix)
        ([[1], [2,3], [], [4, 5]], [[0], [1, 2], [], [0, 3]], [1, 2, 0, 2])
    '''
    matrix = convert_to_scipy_csr(matrix)

    rows_length = []
    count_row_length = 0
    values = []
    columns = []
    for idx_row in xrange(matrix.shape[0]):
        row = matrix.getrow(idx_row)
        row_values = row.data.tolist()
        row_columns = row.indices.tolist()
        count_row_length = len(row_values)

        values.append(row_values)
        columns.append(row_columns)
        rows_length.append(count_row_length)
    return (values, columns, rows_length)


def convert_to_ellpack(matrix, array=True):
    '''
    Method converts a matrix to a format ELLPACK-R.
    For more information about format see article F. Vazquez and other
    entitled: "The sparse matrix vector product on GPUs" [14.06.2009].

    Parameters
    ==========
    matrix : numpy array or scipy matrix
        Input matrix
    array : boolean
        If true, each list in return will be packaged in a numpy array.
        Else will be returned to the normal python list

    Returns
    =======
    converted matrix : tuple of list or tuple of numpy array
        First list is list of values, list of float or numpy.float32.
        Second is list of columns indices, list of integers or numpy.int32.
        Third is list of rows length, list of integers or numpy.int32.

    Examples
    ========
    >>> import numpy
    >>> matrix = numpy.array([[1, 0, 0, 0],
    ...                       [0, 2, 3, 0],
    ...                       [0, 0, 0, 0],
    ...                       [4, 0, 0, 5]])
    >>> convert_to_ellpack(matrix, array=False)
        ([1, 2, 4, 3, 5], [0, 1, 0, 2, 3], [1, 2, 0, 2])
    '''
    matrix = preconvert_to_ell(matrix)
    rows_length = matrix[2]

    values = lu.columns_to_list(lu.normalize_length(matrix[0]))
    cols_indices = lu.columns_to_list(lu.normalize_length(matrix[1]))

    if array == True:
        return (numpy.array(values, dtype=numpy.float32), \
                numpy.array(cols_indices, dtype=numpy.int32), \
                numpy.array(rows_length, dtype=numpy.int32))
    else:
        return (values, cols_indices, rows_length)

def convert_to_sliced(matrix, threads_per_row=2, slice_size=2,
                      align=8, array=True):
    '''
    Method converts a matrix to a format SLICED ELLPACK.
    For more information about format see article A. Dziekonski and other
    entitled: "A memory efficient and fast sparse matrix vector product on
    a gpu" [2011].

    Parameters
    ==========
    matrix : numpy array or scipy matrix
        Input matrix
    array : boolean
        If true, each list in return will be packaged in a numpy array.
        Else will be returned to the normal python list
    threads_per_row : integer
        Threads per row.
    slice_size : integer
        Size of a single slice.
    align : integer
        Constant to calculation and set align of the matrix

    Returns
    =======
    converted matrix : tuple of list or tuple of numpy array
        First list is list of values, list of float or numpy.float32.
        Second is list of columns indices, list of integers or numpy.int32.
        Third is list of rows length, list of integers or numpy.int32.
        Fourth is list of index slices start, list of integers or numpy.int32.

    Examples
    ========
    >>> import numpy
    >>> matrix = numpy.array([[1, 0, 0, 0],
    ...                       [0, 2, 3, 0],
    ...                       [0, 0, 0, 0],
    ...                       [4, 0, 0, 5]])
    >>> convert_to_sliced(matrix, threads_per_row=2, slice_size=2,
    ...                   align=2, array=False)
    ([1, 0, 2, 3, 0, 0, 4, 5], [0, 0, 1, 2, 0, 0, 0, 3], [1, 2, 0, 2],
    [0, 4, 8])
    '''
    matrix = preconvert_to_ell(matrix)
    reshape_ell_to_multiple(matrix, slice_size)
    values = []
    columns_indices = []
    rows_length = matrix[2]
    slices_start = [0, ]

    for group_rows in lu.grouped(matrix[0], slice_size):
        group_rows = lu.normalize_length(group_rows, threads_per_row)
        values.extend(lu.columns_to_list(group_rows, threads_per_row))
        slices_start.append(len(values))
    for group_rows in lu.grouped(matrix[1], slice_size):
        group_rows = lu.normalize_length(group_rows, threads_per_row)
        columns_indices.extend(lu.columns_to_list(group_rows, threads_per_row))

    set_align((values, columns_indices, rows_length, slices_start), align)

    if array == True:
        return (numpy.array(values, dtype=numpy.float32), \
                numpy.array(columns_indices, dtype=numpy.int32), \
                numpy.array(rows_length, dtype=numpy.int32), \
                numpy.array(slices_start, dtype=numpy.int32))
    else:
        return (values, columns_indices, rows_length, slices_start)

def convert_to_sertilp(matrix, threads_per_row, slice_size, prefetch,
                       align=64, array=True):
    '''
    Method converts a matrix to a format SERTILP. Sertilp is
    a format derived from Sliced ​​Ellpack.

    Parameters
    ==========
    matrix : numpy array or scipy matrix
        Input matrix
    threads_per_row : integer
        Threads per row.
    slice_size : integer
        Size of a single slice.
    align : integer
        Constant to calculation and set align of the matrix
    prefetch : integer
        Number of requests for access to data notified in advance.
    array : boolean
        If true, each list in return will be packaged in a numpy array.
        Else will be returned to the normal python list

    Returns
    =======
    converted matrix : tuple of list or tuple of numpy array
        First list is list of values, list of float or numpy.float32.
        Second is list of columns indices, list of integers or numpy.int32.
        Third is list of rows length, list of integers or numpy.int32.
        Fourth is list of index slices start, list of integers or numpy.int32.

    Notes
    =====
    Method authored Krzysztof Sopyła from KMLib:
    https://github.com/ksirg/KMLib . Own translation into Python.

    Examples
    ========
    >>> import numpy
    >>> matrix = numpy.array([[1, 0, 0, 0],
    ...                       [0, 2, 3, 0],
    ...                       [0, 0, 0, 0],
    ...                       [4, 0, 0, 5]])
    >>> convert_to_sertilp(matrix, threads_per_row=2, slice_size=2,
    ...                   align=2, prefetch=2, array=False)
    ([1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0],
     [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
     [1, 1, 0, 1],
     [0, 8, 16])
    '''
    matrix = convert_to_scipy_csr(matrix)

    align = int(ceil(1.0 * slice_size * threads_per_row / align) * align)
    num_rows = matrix.shape[0]
    num_slices = int(ceil((num_rows + 0.0) / slice_size))
    row_length = [0, ] * num_rows
    slices_start = [0, ] * (num_slices + 1)
    slice_max = [0, ] * num_slices
    slice_nr = 0

    for i in range(num_slices):
        slice_max[i] = -1
        idx = -1
        for j in range(slice_size):
            idx = j + i * slice_size
            if idx < num_rows:
                row_length[idx] = matrix.getrow(idx).getnnz()
                if slice_max[i] < row_length[idx]:
                    slice_max[i] = row_length[idx]
        slices_start[i+1] = slices_start[i] + int(ceil(1.0*slice_max[i] \
                            / (prefetch * threads_per_row)) \
                            * prefetch * align)
    nnz_el = slices_start[num_slices]
    vec_cols = [0,] * nnz_el
    vec_vals = [0,] * nnz_el
    slice_nr = 0
    row_in_slice = 0
    for i in range(num_rows):
        slice_nr = i / slice_size
        row_in_slice = i % slice_size
        vec = matrix.getrow(i)

        thread_nr = -1
        value = 0
        col = -1

        row_slice = -1
        for k in range(vec.getnnz()):
            thread_nr = k % threads_per_row
            row_slice = k / threads_per_row
            value = vec.data[k]
            col = vec.indices[k]
            idx = slices_start[slice_nr] + align * row_slice \
                  + row_in_slice * threads_per_row + thread_nr

            vec_vals[idx] = value
            vec_cols[idx] = col

    if array == True:
        return (numpy.array(vec_vals, dtype=numpy.float32), \
            numpy.array(vec_cols, dtype=numpy.int32), \
            numpy.array(row_length, dtype=numpy.int32), \
            numpy.array(slices_start, dtype=numpy.int32))
    else:
        return (vec_vals, vec_cols, row_length, slices_start)

def convert_to_ertilp(matrix, prefetch, threads_per_row, array=True):
    '''
    Method converts a matrix to a format ERTILP. Sertilp is
    a format derived from ​​Ellpack.

    Parameters
    ==========
    matrix : numpy array or scipy matrix
        Input matrix
    threads_per_row : integer
        Threads per row.
    prefetch : integer
        Number of requests for access to data notified in advance.
    array : boolean
        If true, each list in return will be packaged in a numpy array.
        Else will be returned to the normal python list

    Returns
    =======
    converted matrix : tuple of list or tuple of numpy array
        First list is list of values, list of float or numpy.float32.
        Second is list of columns indices, list of integers or numpy.int32.
        Third is list of rows length, list of integers or numpy.int32.

    Notes
    =====
    Method authored Krzysztof Sopyła from KMLib:
    https://github.com/ksirg/KMLib . Own translation into Python.

    Examples
    ========
    >>> import numpy
    >>> matrix = numpy.array([[1, 0, 0, 0],
    ...                       [0, 2, 3, 0],
    ...                       [0, 0, 0, 0],
    ...                       [4, 0, 0, 5]])
    >>> convert_to_ertilp(matrix, threads_per_row=2, prefetch=2,
        array=False)
    ([1, 0, 2, 3, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 1])
    '''
    matrix = convert_to_scipy_csr(matrix)
    align = threads_per_row * prefetch

    max_el = 1
    max_el = max([m.getnnz() for m in matrix])
    rest = max_el % align

    if rest != 0:
        max_el = max_el + align - rest

#    avgEl = mean([m.getnnz() for m in matrix])
    num_rows = matrix.shape[0]
    vec_vals = [0, ] * (num_rows * max_el)
    vec_cols = [0, ] * (num_rows * max_el)
    row_length = [0, ] * num_rows

    for i in range(num_rows):
        vec = matrix.getrow(i)
        for j in range(vec.getnnz()):
            k = j / threads_per_row
            t = j % threads_per_row
            vec_vals[k * num_rows * threads_per_row \
                     + i * threads_per_row + t] = vec.data[j]
            vec_cols[k * num_rows * threads_per_row \
                     + i * threads_per_row + t] = vec.indices[j]
        row_length[i] = int(vec.getnnz())

    if array == True:
        return (numpy.array(vec_vals, dtype=numpy.float32), \
            numpy.array(vec_cols, dtype=numpy.int32), \
            numpy.array(row_length, dtype=numpy.int32))
    else:
        return (vec_vals, vec_cols, row_length)

def convert_to_scipy_csr(matrix):
    '''
    Method converts a matrix to a format CSR.
    For more information about format see Scipy documentation.

    Parameters
    ==========
    matrix : numpy array or scipy matrix
        Matrix to convertion

    Returns
    =======
    converted matrix : scipy.sparse.csr_matrix
    '''
    if scipy.sparse.isspmatrix_csr(matrix):
        return matrix
    elif scipy.sparse.isspmatrix(matrix):
        return matrix.tocsr()
    elif isinstance(matrix, numpy.ndarray):
        return scipy.sparse.csr_matrix(matrix)
    else:
        raise NotImplementedError('This matrix type is not supported.'
                                  'Only support numpy.ndarray and'
                                  'scipy.sparse matrix.')

if __name__ == "__main__":
    pass
                     