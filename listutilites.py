# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:57:18 2014
@author: Sławomir Figiel

Module have method work in lists and list of list.
"""
from itertools import izip, chain

def normalize_length(list_of_list, multiple=1):
    '''
    It normalizes the length of all the lists from the list of lists passed
    as an argument.

    Length of the lists will be the nearest multiple of the argument
    "multiple", not less than the length of the longest of them.

    Parameters
    ==========
    list_of_list : enumerate
        enumerates in enumerate
    multiple : int > 0
        length of lists is a multiple parameter

    Returns
    =======
    out : list of list
        The original list of added zeros

    Notes
    =====
    List of list can not be empty. However, it can contain empty list.

    Examples
    ========
    >>> list_of_list = [[3, 1], [1], [1, 2, 3]]
    >>> normalize_length(list_of_list, 4)
        [[3, 1, 0, 0], [1, 0, 0, 0], [1, 2, 3, 0]]
    '''
    length_max = max([len(i) for i in list_of_list])
    rest = length_max % multiple
    if rest:
        length_max += multiple - rest
    return [l + [0, ] * (length_max - len(l)) for l in list_of_list]

def columns_to_list(list_of_list, group_size=1):
    '''
    Returns a list of list_of_list read columns. GROUP_SIZE parameter
    specifies how many values ​​will be read at one time in a row.

    Parameters
    ==========
    list_of_list : enumerate
        enumerates in enumerate
    group_size : int > 0
        size of group value in row

    Returns
    =======
    out : list
        List of value from list_of_list

    Notes
    =====
    The length of each list must be normalized. Their length must
    be a multiple of group_size.

    Examples
    ========
    >>> list_of_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    >>> columns_to_list(list_of_list, 2)
        [1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12]
    '''
    groups = [grouped(i, group_size) for i in list_of_list]
    cols_list = izip(*groups)
    return list(chain.from_iterable(chain.from_iterable(cols_list)))

def grouped(iterable, items_in_group):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,..."
    return izip(*[iter(iterable)]*items_in_group)

if __name__ == '__main__':
    pass
                        