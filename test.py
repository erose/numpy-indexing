import unittest
import numpy as np
import numpy.testing

import numpy_indexing

# Python converts an indexing expression like A[1, :, 2] to (1, slice(None), 2) in order to pass it
# to __getitem__, but it does so under the hood. Especially because of ':', we need something to
# implement __getitem__ in order to give us access to this conversion.
class WrapperToTestIndexingExpressions:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, obj):
        return numpy_indexing.shape_after_indexing(self.array.shape, obj)

class TestShapeAfterIndexing(unittest.TestCase):
    def assertAgreesWithNumpy(self, array, indexing_string):
        string = 'A' + indexing_string # so it becomes something like 'A[0, 1]'

        actual = eval(string, {}, {'np': np, 'A': array}).shape
        expected = eval(string, {}, {'np': np, 'A': WrapperToTestIndexingExpressions(array)})
        self.assertEqual(expected, actual)

    def test_empty_slices(self):
        A = np.array([[1, 2, 3], [2, 4, 6]])
        self.assertAgreesWithNumpy(A, '[:]')
        self.assertAgreesWithNumpy(A, '[:, :]')

    def test_slices(self):
        A = np.array([[1, 2, 3], [2, 4, 6]])
        self.assertAgreesWithNumpy(A, '[0:1]')
        self.assertAgreesWithNumpy(A, '[:, 1:2]')
        self.assertAgreesWithNumpy(A, '[:2]')
        self.assertAgreesWithNumpy(A, '[1:]')
        self.assertAgreesWithNumpy(A, '[:10]')

    def test_integers(self):
        A = np.array([[1, 2, 3], [2, 4, 6]])
        self.assertAgreesWithNumpy(A, '[0, 1]')
        self.assertAgreesWithNumpy(A, '[1]')
        self.assertAgreesWithNumpy(A, '[0, :]')
        self.assertAgreesWithNumpy(A, '[:, 1]')

    def test_lists(self):
        A = np.array([[1, 2, 3], [2, 4, 6]])
        self.assertAgreesWithNumpy(A, '[[0]]')
        self.assertAgreesWithNumpy(A, '[[0], [1]]')
        self.assertAgreesWithNumpy(A, '[[0, 1, 0], [1, 1, 1]]')

    def test_newaxis(self):
        A = np.array([1, 2, 3])
        self.assertAgreesWithNumpy(A, '[np.newaxis, 0]')
        self.assertAgreesWithNumpy(A, '[0, np.newaxis]')

    def test_mixture_on_3d_array(self):
        A = np.array(
            [
                [[ 1,  2,  3], [ 2,  4,  6]],
                [[-1, -2, -3], [-2, -4, -6]],
            ]
        )

        self.assertAgreesWithNumpy(A, '[[0], :]')
        self.assertAgreesWithNumpy(A, '[[0, 1], :]')
        self.assertAgreesWithNumpy(A, '[[0, 1], 1:2]')
        self.assertAgreesWithNumpy(A, '[[0, 1], [1, 1], :]')
        self.assertAgreesWithNumpy(A, '[[0, 1], :, [1, 1]]')
        self.assertAgreesWithNumpy(A, '[1:2, [1, 1]]')

    def test_mixture_on_4d_array(self):
        A = np.array(
            [
                [
                    [[1, 2, 3 ], [10, 20, 30 ]]
                ],
                [
                    [[2, 4, 6 ], [20, 40, 60 ]]
                ],
                [
                    [[3, 6, 9 ], [30, 60, 90 ]]
                ],
                [
                    [[4, 8, 12], [40, 80, 120]]
                ],
            ]
        )

        self.assertAgreesWithNumpy(A, '[[0, 1], :, :, [0, 0]]')
        self.assertAgreesWithNumpy(A, '[:, :, :, [0, 0]]')
        self.assertAgreesWithNumpy(A, '[[0, 0], :, :, :]')
        self.assertAgreesWithNumpy(A, '[[0, 1], [0, 0], :]')
        self.assertAgreesWithNumpy(A, '[[0, 1], [0, 0], :, 2]')
        self.assertAgreesWithNumpy(A, '[[0, 1], :, [1, 1], :]')

if __name__ == "__main__":
    unittest.main()
