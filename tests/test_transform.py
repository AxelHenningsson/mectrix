import unittest
import numpy as np
import mectrix

class TestTransform(unittest.TestCase):

    def setUp(self):
        pass

    def test_voigt_to_stress_tensor(self):
        stress_voigt = np.array([[1, 2, 3, 4, 5, 6], [-1, -1, -1, -1, -1, -1]]).T
        stress_tensor =  mectrix.transforms.voigt_to_stress_tensor(stress_voigt)

        self.assertTrue(stress_tensor.shape == (3, 3, 2))
        self.assertTrue( np.allclose(stress_tensor[:,:,0] , np.array([ [1, 6, 5],
                                                                       [6, 2, 4],
                                                                       [5, 4, 3] ])) )
        self.assertTrue( np.allclose(stress_tensor[:,:,1] , -np.ones((3,3))) )

    def test_voigt_to_strain_tensor(self):
        strain_voigt = np.array([[1, 2, 3, 4*2., 5*2., 6*2.], [-1, -1, -1, -1*2., -1*2., -1*2.]]).T
        strain_tensor =  mectrix.transforms.voigt_to_strain_tensor(strain_voigt)

        self.assertTrue(strain_tensor.shape==(3, 3, 2))
        self.assertTrue( np.allclose(strain_tensor[:,:,0] , np.array([ [1, 6, 5],
                                                                       [6, 2, 4],
                                                                       [5, 4, 3] ])) )
        self.assertTrue( np.allclose(strain_tensor[:,:,1] , -np.ones((3,3))) )

    def test_strain_tensor_to_voigt(self):
        strain_voigt = np.array([[1, 2, 3, 4*2., 5*2., 6*2.], [-1, -1, -1, -1*2., -1*2., -1*2.]]).T
        strain_tensor =  mectrix.transforms.voigt_to_strain_tensor(strain_voigt)
        strain_voigt2 =  mectrix.transforms.strain_tensor_to_voigt(strain_tensor)
        self.assertTrue( np.allclose(strain_voigt, strain_voigt2) )

    def test_stress_tensor_to_voigt(self):
        stress_voigt = np.array([[1, 2, 3, 4*2., 5*2., 6*2.], [-1, -1, -1, -1*2., -1*2., -1*2.]]).T
        stress_tensor =  mectrix.transforms.voigt_to_strain_tensor(stress_voigt)
        stress_voigt2 =  mectrix.transforms.strain_tensor_to_voigt(stress_tensor)
        self.assertTrue( np.allclose(stress_voigt, stress_voigt2) )

    def test_get_elasticity_rotation_matrix(self):
        theta = np.pi / 4
        R1 = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
        theta = np.pi / 8
        R2 = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
        rotation_matrix = np.array([R1,R2]).reshape(3,3,2)

        M = mectrix.transforms.get_elasticity_rotation_matrix(rotation_matrix)
        self.assertTrue(M.shape == (6, 6, 2))

    def test_get_compliance_rotation_matrix(self):
        theta = np.pi / 4
        R1 = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
        theta = np.pi / 8
        R2 = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
        rotation_matrix = np.array([R1,R2]).reshape(3,3,2)

        N = mectrix.transforms.get_compliance_rotation_matrix(rotation_matrix)
        self.assertTrue(N.shape == (6, 6, 2))

    def test_rotate_elasticity_matrix(self):
        theta1 = np.pi / 4
        r1 = np.array([
                    [np.cos(theta1), -np.sin(theta1), 0],
                    [np.sin(theta1), np.cos(theta1), 0],
                    [0, 0, 1]
                ])
        theta2 = np.pi / 6
        r2 = np.array([
            [1, 0, 0],
            [0, np.cos(theta2), -np.sin(theta2)],
            [0, np.sin(theta2), np.cos(theta2)]
        ])

        rotation_matrix = r1 @ r2

        C11, C33, C44, C66, C12, C13, C14 = 86.99, 106.39, 58.12, 40.12, 6.75, 12.17, 17.99
        C = np.zeros((6, 6))
        C[0, :4] = [C11, C12, C13, C14]
        C[1, 1:4] = [C11, C13, -C14]
        C[2, 2] = C33
        C[3, 3] = C44
        C[4, 4:6] = [C44, C14]
        C[5, 5] = C66
        C = C + np.tril(C.T, -1)

        # alternative 1
        strain_voigt = np.array([1, 2.1, 3, 4, -1, -2])
        C_prime = mectrix.transforms.rotate_elasticity_matrix(C, rotation_matrix)
        stress_voigt = C_prime @ strain_voigt
        stress_tensor1 = mectrix.transforms.voigt_to_stress_tensor(stress_voigt)

        # alternative 2
        strain_tensor = mectrix.transforms.voigt_to_strain_tensor(strain_voigt)
        strain_tensor_rotated = rotation_matrix.T @ strain_tensor @ rotation_matrix
        strain_voigt_rotated = mectrix.transforms.strain_tensor_to_voigt(strain_tensor_rotated)
        stress_voigt_rotated = C @ strain_voigt_rotated
        stress_tensor_rotated = mectrix.transforms.voigt_to_stress_tensor(stress_voigt_rotated)
        stress_tensor2 = rotation_matrix @ stress_tensor_rotated @ rotation_matrix.T

        self.assertTrue( np.allclose(stress_tensor1, stress_tensor2) )


    def test_rotate_compliance_matrix(self):
        theta1 = np.pi / 4
        r1 = np.array([
                    [np.cos(theta1), -np.sin(theta1), 0],
                    [np.sin(theta1), np.cos(theta1), 0],
                    [0, 0, 1]
                ])
        theta2 = np.pi / 6
        r2 = np.array([
            [1, 0, 0],
            [0, np.cos(theta2), -np.sin(theta2)],
            [0, np.sin(theta2), np.cos(theta2)]
        ])

        rotation_matrix = r1 @ r2

        C11, C33, C44, C66, C12, C13, C14 = 86.99, 106.39, 58.12, 40.12, 6.75, 12.17, 17.99
        C = np.zeros((6, 6))
        C[0, :4] = [C11, C12, C13, C14]
        C[1, 1:4] = [C11, C13, -C14]
        C[2, 2] = C33
        C[3, 3] = C44
        C[4, 4:6] = [C44, C14]
        C[5, 5] = C66
        C = C + np.tril(C.T, -1)
        S = np.linalg.inv(C)

        # alternative 1
        stress_voigt = np.array([1, 2.1, 3, 4, -1, -2])
        S_prime = mectrix.transforms.rotate_compliance_matrix(S, rotation_matrix)
        strain_voigt = S_prime @ stress_voigt
        strain_tensor1 = mectrix.transforms.voigt_to_strain_tensor(strain_voigt)

        # alternative 2
        stress_tensor = mectrix.transforms.voigt_to_stress_tensor(stress_voigt)
        stress_tensor_rotated = rotation_matrix.T @ stress_tensor @ rotation_matrix
        stress_voigt_rotated = mectrix.transforms.stress_tensor_to_voigt(stress_tensor_rotated)
        strain_voigt_rotated = S @ stress_voigt_rotated
        strain_tensor_rotated = mectrix.transforms.voigt_to_strain_tensor(strain_voigt_rotated)
        strain_tensor2 = rotation_matrix @ strain_tensor_rotated @ rotation_matrix.T

        self.assertTrue( np.allclose(strain_tensor1, strain_tensor2) )


if __name__ == '__main__':
    unittest.main()
