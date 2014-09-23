# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:01:10 2014

@author: Sławomir Figiel
"""

import unittest
import matrixFormat

class ListHelpersMethod(unittest.TestCase):
    list_of_list = {
                        1 : [
                                [3, 4, 1],
                                [2, 1, 0, 7, 8],
                                [1,]
                            ],
                        2 : [
                                [],
                            ],
                        3 : [
                                [1,],
                            ],
                        4 : [
                                [-1, -3.0, 4, 'axz', 213, 213, 'dfs'],
                                [],
                                [],
                            ],
                        5 : [
                                [1,],
                                [3,],
                                ['dfd']
                            ]
                    }
    def testNormalizationResult(self):
        '''normalizeLength should return the same list.'''
        right_value = {
                        1 : [
                                [3, 4, 1, 0, 0],
                                [2, 1, 0, 7, 8],
                                [1, 0, 0, 0, 0]
                            ],
                        2 : [
                                [],
                            ],
                        3 : [
                                [1,],
                            ],
                        4 : [
                                [-1, -3.0, 4, 'axz', 213, 213, 'dfs'],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]
                            ],
                        5 : [
                                [1,],
                                [3,],
                                ['dfd']
                            ]
                    }
            
        for k, v in right_value.items():
            self.assertEqual(matrixFormat.normalizujDlugosci(self.list_of_list[k], 1), v)
        
    def testNormalizationLengthMax(self):
        right_value = {
                        1 : 5,
                        2 : 0,
                        3 : 1,
                        4 : 7,
                        5 : 1
                      }
        for k, v in right_value.items():
            list_of_list = self.list_of_list[k]
            normalizated = matrixFormat.normalizujDlugosci(list_of_list, 1)
            maximum = max(len(i) for i in normalizated)
            self.assertEqual(maximum, v)
    def testNormalizationLengthEqual(self):
        for v in self.list_of_list.values():
            result = matrixFormat.normalizujDlugosci(v, 1)
            length = len(result[0])
            for l in result:
                self.assertEqual(length, len(l))
    def testNormalizationMultiple(self):
        testList = [
                        [3, 4, 1],
                        [2, 1, 0, 7, 8],
                        [1,]
                    ]
        right_value = {
                        4 : [
                                [3, 4, 1, 0, 0, 0, 0, 0],
                                [2, 1, 0, 7, 8, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0]
                            ],
                        5 : [
                                [3, 4, 1, 0, 0],
                                [2, 1, 0, 7, 8],
                                [1, 0, 0, 0, 0]
                            ]
                      }
        right_value.update(dict.fromkeys([2, 3, 6],
                            [
                                [3, 4, 1, 0, 0, 0],
                                [2, 1, 0, 7, 8, 0],
                                [1, 0, 0, 0, 0, 0]
                            ]))
        for k, v in right_value.items():
            self.assertEqual(matrixFormat.normalizujDlugosci(testList, k), v)
            
if __name__ == "__main__":
    unittest.main()

        