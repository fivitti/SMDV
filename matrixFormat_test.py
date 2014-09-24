# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:23:11 2014

@author: Sławomir Figiel
"""

import unittest
import copy
import matrixFormat

class helpersMethod(unittest.TestCase):
    def testReshape(self):
        matrix = [
                    [[1], [3, 1], [7]],
                    [[1], [0, 2], [2]],
                    [1, 2, 1]
                ]
        right_result_3 = matrix
        right_result_4 = [
                            [[1], [3, 1], [7], []],
                            [[1], [0, 2], [2], []],
                            [1, 2, 1, 0]
                        ]
        right_result_5 = [
                            [[1], [3, 1], [7], [], []],
                            [[1], [0, 2], [2], [], []],
                            [1, 2, 1, 0, 0]
                        ]
        right_result_2 = right_result_4
        right_result = [(right_result_2, 2), (right_result_3, 3), 
                        (right_result_4, 4), (right_result_5, 5)]
        for right_matrix, mult in right_result:
            temp_matrix = copy.deepcopy(matrix)
            matrixFormat.reshape_to_multiple_ell(temp_matrix, mult)
            self.assertEqual(
                temp_matrix,
                right_matrix
            )
if __name__ == "__main__":
    unittest.main()