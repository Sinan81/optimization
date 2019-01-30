#!/usr/bin/env python

import q1
import unittest
import numpy as np

class test_functional(unittest.TestCase):

    def setUp(self):
        pass


    def test_H(self):
        xi = np.array([1, 2, 3])
        xj = np.array([4, 5, 6])
        qij = np.array([7, 8, 9])

        self.assertEqual( q1.H(qij,xi,xj), 270)

    def test_set_up_x(self):
        ia = np.array([0, 1, 2])
        iju = np.array([0, 1, 2, 3])

        # case with xi label index '0'
        config = np.array([1, 0, 0, 0])
        self.assertSequenceEqual( q1.set_up_x(ia,iju,config).tolist(), np.array([1, 0, 0]).tolist() )
        # toggle all xi to '1'
        config = np.array([1, 1, 1, 1])
        self.assertSequenceEqual( q1.set_up_x(ia,iju,config).tolist(), np.array([1, 1, 1]).tolist() )

    def test_calc_H_exact(self):
        # test for the case of a "toy" system with 3-decision variables
        test_input = np.array([[1, 1, -2.0], [3, 3, 3.0], [2, 3, -1.0]])
        np.savetxt('input.txt', test_input, fmt="%2.3g")
        expected_output = np.array([[ 4., -2.,  1.,  0.,  0.],
                                    [ 6., -2.,  1.,  1.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.],
                                    [ 2.,  0.,  0.,  1.,  0.],
                                    [ 7.,  0.,  1.,  1.,  1.],
                                    [ 5.,  1.,  1.,  0.,  1.],
                                    [ 3.,  2.,  0.,  1.,  1.],
                                    [ 1.,  3.,  0.,  0.,  1.]])

        self.assertSequenceEqual( q1.calc_H_exact().tolist(), expected_output.tolist() )


if __name__=='__main__':
    unittest.main()
