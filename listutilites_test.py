# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:01:10 2014

@author: Sławomir Figiel
"""

import unittest
import listutilites

class NormalizationLength(unittest.TestCase):
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
            self.assertEqual(listutilites.normalize_length(self.list_of_list[k], 1), v)
        
    def testNormalizationLengthMax(self):
        '''Checks the maximum length of the list from the list of lists.'''
        right_value = {
                        1 : 5,
                        2 : 0,
                        3 : 1,
                        4 : 7,
                        5 : 1
                      }
        for k, v in right_value.items():
            list_of_list = self.list_of_list[k]
            normalizated = listutilites.normalize_length(list_of_list, 1)
            maximum = max(len(i) for i in normalizated)
            self.assertEqual(maximum, v)
    def testNormalizationLengthEqual(self):
        '''The lengths of all lists should be equal.'''
        for v in self.list_of_list.values():
            result = listutilites.normalize_length(v, 1)
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
            self.assertEqual(listutilites.normalize_length(testList, k), v)

class ColumnToList(unittest.TestCase):
    def testConvertList(self):
        '''Lists should be correctly converted.'''
        list_of_list = [
                        [
                            [2, 3, 1],
                            [1, 2, 7]
                        ],
                        [
                            [],
                            [],
                            [],
                            []
                        ],
                        [
                            ['aaa', 'bbb', 'ccc'],
                            [-1, 8.32232, []]
                        ]
                    ]
        right_value = [
                        [2, 1, 3, 2, 1, 7],
                        [],
                        ['aaa', -1, 'bbb', 8.32232, 'ccc', []]
                    ]
        test_set = zip(list_of_list, right_value)
        for l, rv in test_set:
            self.assertEqual(listutilites.columns_to_list(l), rv)
            
    def testConvertListGroup(self):
        '''Lists should be correctly converted.'''
        list_of_list = [
                        [
                            [2, 3, 1, 0, 9, 4],
                            [1, 2, 7, 2, 1, 2]
                        ],
                        [
                            [1, 0],
                            [3, 3],
                            [4, 4],
                            [5, 5]
                        ],
                        [
                            ['aaa', 'bbb', 'ccc'],
                            [-1, 8.32232, []]
                        ]
                    ]
        right_value = [
                        [2, 3, 1, 1, 2, 7, 0, 9, 4, 2, 1, 2],
                        [1, 0, 3, 3, 4, 4, 5, 5],
                        ['aaa', 'bbb', -1, 8.32232]
                    ]
        group_sizes = [3, 2, 2]  
        test_set = zip(list_of_list, right_value, group_sizes)
        for l, rv, gs in test_set:
            self.assertEqual(listutilites.columns_to_list(l, gs), rv)                      
if __name__ == "__main__":
    unittest.main()

        