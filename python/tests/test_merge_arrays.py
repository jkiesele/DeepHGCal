import numpy as np
from libs.merge import merge_two_arrays, merge_two_arrays_separate
import unittest


class MergeTestSmall(unittest.TestCase):
    def setUp(self):
        self.A = np.array(
            [
                # Batch element 0
                [
                    # Entry 0
                    [10, 3, 3],
                    # Entry 1
                    [20, 4, 4],
                    # Entry 2
                    [30, 5, 5],
                ],
                # Batch element 1
                [
                    # Entry 0
                    [10, 3, 3],
                    # Entry 1
                    [20, 4, 4],
                    # Entry 2
                    [0, 0, 0],
                ]
            ]
            , dtype=np.float32)

        self.B = np.array(
            [
                # Batch element 0
                [
                    # Entry 0
                    [11, 3, 3],
                    # Entry 1
                    [21, 4, 4],
                    # Entry 2
                    [40, 6, 6],
                ],
                # Batch element 1
                [
                    # Entry 0
                    [10, 3, 3],
                    # Entry 1
                    [20, 4, 4],
                    # Entry 2
                    [0, 0, 0],
                ]
            ]
            , dtype=np.float32)

        self.idsA = np.array(
            [
                # Batch element 0
                [
                    0,
                    1,
                    2
                ],
                # Batch element 1
                [
                    1,
                    2,
                    0
                ]
            ]
            , dtype=np.int32)

        self.idsB = np.array(
            [
                # Batch element 0
                [
                    0,
                    1,
                    3
                ],
                # Batch element 1
                [
                    1,
                    2,
                    0
                ]
            ]
            , dtype=np.int32)

        self.sizesA = np.array(
            [
                3, 2
            ]
            , dtype=np.int32)

        self.sizesB = np.array(
            [
                3, 2
            ]
            , dtype=np.int32)

        self.dataResult = np.array([[
                                        [21., 3., 3.],
                                        [41., 4., 4.],
                                        [30., 5., 5.],
                                        [40., 6., 6.],
                                        [0., 0., 0.],
                                        [0., 0., 0.]],
                                    [
                                        [20., 3., 3.],
                                        [40., 4., 4.],
                                        [0., 0., 0.],
                                        [0., 0., 0.],
                                        [0., 0., 0.],
                                        [0., 0., 0.]
                                    ]
                                    ], dtype=np.float32)
        self.dataResultSeparate = np.array([[[10., 11.,  3.,  3.],
                                        [20., 21.,  4.,  4.],
                                        [30.,  0.,  5.,  5.],
                                        [ 0., 40.,  6.,  6.],
                                        [ 0.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  0.]],

                                       [[10., 10.,  3.,  3.],
                                        [20., 20.,  4.,  4.],
                                        [ 0.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  0.]]], dtype=np.float32)


    def testEqual(self):
        data, ids, sizes = merge_two_arrays(self.A, self.B, self.idsA, self.idsB, self.sizesA, self.sizesB)

        assert np.allclose(data, self.dataResult)

    def testEqualSeparate(self):
        data, ids, sizes = merge_two_arrays_separate(self.A, self.B, self.idsA, self.idsB, self.sizesA, self.sizesB)

        assert np.allclose(data, self.dataResultSeparate)



if __name__ == '__main__':
    unittest.main()
