import unittest
import numpy as np
import mectrix


class TestTransform(unittest.TestCase):


    def setUp(self):
        pass


    def test_stress_tensor_to_voigt(self):
        stress_voigt = np.array([[1, 2, 3, 4, 5, 6], [-1, -1, -1, -1, -1, -1]]).T
        stress_tensor =  mectrix.transforms.voigt_to_stress_tensor(stress_voigt)

        self.assertTrue(stress_tensor.shape==(3, 3, 2))
        self.assertTrue( np.allclose(stress_tensor[:,:,0] , np.array([ [1, 6, 5],
                                                                       [6, 2, 4],
                                                                       [5, 4, 3] ])) )
        self.assertTrue( np.allclose(stress_tensor[:,:,1] , -np.ones((3,3))) )


    def test_strain_tensor_to_voigt(self):
        strain_voigt = np.array([[1, 2, 3, 4*2., 5*2., 6*2.], [-1, -1, -1, -1*2., -1*2., -1*2.]]).T
        strain_tensor =  mectrix.transforms.voigt_to_strain_tensor(strain_voigt)

        self.assertTrue(strain_tensor.shape==(3, 3, 2))
        self.assertTrue( np.allclose(strain_tensor[:,:,0] , np.array([ [1, 6, 5],
                                                                       [6, 2, 4],
                                                                       [5, 4, 3] ])) )
        self.assertTrue( np.allclose(strain_tensor[:,:,1] , -np.ones((3,3))) )


    def test_voigt_to_strain_tensor(self):
        pass


    def test_voigt_to_stress_tensor(self):
        pass

if __name__ == '__main__':
    unittest.main()
