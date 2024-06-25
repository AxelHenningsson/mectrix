import numpy as np


def stress_tensor_to_voigt(stress_tensor):
    """Convert a 3x3 stress tensor to a flat array following Voigt notation.

    The components of the input stress_tensor tensor are:
            [[sigma_xx, sigma_xy, sigma_xz],
             [sigma_xy, sigma_yy, sigma_yz],
             [sigma_xz, sigma_yz, sigma_zz]].

    Defenition of Voigt notation: https://en.wikipedia.org/wiki/Voigt_notation

    Args:
        stress_tensor (:obj:`numpy.array`): The stress tensor, shape=(3,3,N).

    Returns:
        :obj:`numpy.array`: stress tensor in Voigt notation, shape=(6,N).
            The components are ordered as: eps_xx, eps_yy, eps_zz, 2*eps_yz, 2*eps_xz, 2*eps_xy
    """
    stress_tensor_in_voigt_notation = np.array([
                                            stress_tensor[0, 0],  # sigma_xx
                                            stress_tensor[1, 1],  # sigma_yy
                                            stress_tensor[2, 2],  # sigma_zz
                                            stress_tensor[1, 2],  # sigma_yz = sigma_zy
                                            stress_tensor[0, 2],  # sigma_xz = sigma_zx
                                            stress_tensor[0, 1]   # sigma_xy = sigma_yx
    ])
    return stress_tensor_in_voigt_notation


def strain_tensor_to_voigt(strain_tensor):
    """Convert a 3x3 strain tensor to a flat array following Voigt notation.

    The components of the input strain_tensor tensor are:
            [[epsilon_xx, epsilon_xy, epsilon_xz],
             [epsilon_xy, epsilon_yy, epsilon_yz],
             [epsilon_xz, epsilon_yz, epsilon_zz]].

    Defenition of Voigt notation: https://en.wikipedia.org/wiki/Voigt_notation

    Args:
        strain_tensor (:obj:`numpy.array`): The strain tensor, shape=(3,3,N).

    Returns:
        :obj:`numpy.array`: strain tensor in Voigt notation, shape=(6,N).
            The components are ordered as: sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sig_xy
    """
    strain_tensor_in_voigt_notation = np.array([
                                        strain_tensor[0, 0],      # epsilon_xx
                                        strain_tensor[1, 1],      # epsilon_yy
                                        strain_tensor[2, 2],      # epsilon_zz
                                        2 * strain_tensor[1, 2],  # 2 * epsilon_yz = 2 * epsilon_zy
                                        2 * strain_tensor[0, 2],  # 2 * epsilon_xz = 2 * epsilon_zx
                                        2 * strain_tensor[0, 1]   # 2 * epsilon_xy = 2 * epsilon_yx
                                    ])
    return strain_tensor_in_voigt_notation


def voigt_to_strain_tensor(voigt_notation):
    """Convert a flat array in Voigt notation to a 3x3 strain tensor.

    Defenition of voigt notation is found here https://en.wikipedia.org/wiki/Voigt_notation.

    The components are ordered as:
        voigt_notation = [eps_xx, eps_yy, eps_zz, 2*eps_yz, 2*eps_xz, 2*eps_xy].

    Args:
        voigt_notation (:obj:`numpy.ndarray`): Array in Voigt notation, shape (6,N).

    Returns:
        :obj:`numpy.ndarray`: 3x3 strain tensor, shape (3, 3, N).
            The components of the output strain tensor tensor are:
                    [[epsilon_xx, epsilon_xy, epsilon_xz],
                    [epsilon_xy, epsilon_yy, epsilon_yz],
                    [epsilon_xz, epsilon_yz, epsilon_zz]].
    """
    strain_tensor = np.array([
        [voigt_notation[0], voigt_notation[5] / 2, voigt_notation[4] / 2],
        [voigt_notation[5] / 2, voigt_notation[1], voigt_notation[3] / 2],
        [voigt_notation[4] / 2, voigt_notation[3] / 2, voigt_notation[2]]
    ])
    return strain_tensor


def voigt_to_stress_tensor(voigt_notation):
    """Convert a flat array in Voigt notation to a 3x3 stress tensor.

    Defenition of voigt notation is found here https://en.wikipedia.org/wiki/Voigt_notation.

    The components are ordered as:
        voigt_notation = [sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sig_xy].

    Args:
        voigt_notation (:obj:`numpy.ndarray`): Array in Voigt notation, shape (6,N).

    Returns:
        :obj:`numpy.ndarray`: 3x3 stress tensor, shape (3, 3, N).
            The components of the output stress tensor tensor are:
                [[sigma_xx, sigma_xy, sigma_xz],
                [sigma_xy, sigma_yy, sigma_yz],
                [sigma_xz, sigma_yz, sigma_zz]].
    """
    stress_tensor = np.array([
        [voigt_notation[0], voigt_notation[5], voigt_notation[4]],
        [voigt_notation[5], voigt_notation[1], voigt_notation[3]],
        [voigt_notation[4], voigt_notation[3], voigt_notation[2]]
    ])
    return stress_tensor


def get_elasticity_rotation_matrix(rotation_matrix):
    """Compute the 6x6 transformation matrix that corresponds to a rotation of the elasticity matrix.

    To clarify, given a stiffness matrix, C, and a strain in Voigt notation, the following two
    code snippets will produce equivalent stress tensors:

        # alternative 1
        M = get_elasticity_rotation_matrix(rotation_matrix)
        C_prime = M @ C @ M.T
        stress_voigt = C_prime @ strain_voigt
        stress_tensor = voigt_to_stress_tensor(stress_voigt)

        # alternative 2
        strain_tensor = voigt_to_strain_tensor(strain_voigt)
        strain_tensor_rotated = rotation_matrix.T @ strain_tensor @ rotation_matrix
        strain_voigt_rotated = strain_tensor_to_voigt(strain_tensor_rotated)
        stress_voigt_rotated = C @ strain_voigt_rotated
        stress_tensor_rotated = voigt_to_stress_tensor(stress_voigt_rotated)
        stress_tensor = rotation_matrix @ stress_tensor_rotated @ rotation_matrix.T

    I.e we interpret the rotation_matrix to be the rotation transform that bring a pre-defined cooridnate
    system into the coordinate system of the strain_tensor. C_prime is then the elasticity matrix
    described in the strain_tensor local cooridnate system.

    NOTE: it is often faster to compute C_prime = M @ C @ M.T once and apply it to many strain states compared
    to looping over the above alternative 2.

    Args:
        rotation_matrix (:obj:`numpy.ndarray`): Unitary rotaiton matrix, shape=(6,6,N)

    Returns:
        :obj:`numpy.ndarray`: elasticity rotation matrix, shape=(6,6,N)
    """
    Uxx, Uxy, Uxz = rotation_matrix[0]
    Uyx, Uyy, Uyz = rotation_matrix[1]
    Uzx, Uzy, Uzz = rotation_matrix[2]

    M = np.array([
        [Uxx**2, Uxy**2, Uxz**2, 2*Uxy*Uxz, 2*Uxx*Uxz, 2*Uxx*Uxy],
        [Uyx**2, Uyy**2, Uyz**2, 2*Uyy*Uyz, 2*Uyx*Uyz, 2*Uyx*Uyy],
        [Uzx**2, Uzy**2, Uzz**2, 2*Uzy*Uzz, 2*Uzx*Uzz, 2*Uzx*Uzy],
        [Uyx*Uzx, Uyy*Uzy, Uyz*Uzz, Uyy*Uzz + Uyz*Uzy, Uyx*Uzz + Uyz*Uzx, Uyx*Uzy + Uyy*Uzx],
        [Uxx*Uzx, Uxy*Uzy, Uxz*Uzz, Uxy*Uzz + Uxz*Uzy, Uxx*Uzz + Uxz*Uzx, Uxx*Uzy + Uxy*Uzx],
        [Uxx*Uyx, Uxy*Uyy, Uxz*Uyz, Uxy*Uyz + Uxz*Uyy, Uxx*Uyz + Uxz*Uyx, Uxx*Uyy + Uxy*Uyx]
    ])

    return M


def get_compliance_rotation_matrix(rotation_matrix):
    """Compute the 6x6 transformation matrix that corresponds to a rotation of the compliance matrix.

    To clarify, given a compliance matrix, S, and a stress in Voigt notation, the following two
    code snippets will produce equivalent strain tensors:

        # alternative 1
        N = get_compliance_rotation_matrix(rotation_matrix)
        S_prime = M @ S @ M.T
        strain_voigt = S_prime @ stress_voigt
        straain_tensor = voigt_to_stress_tensor(strain_voigt)

        # alternative 2
        stress_tensor = voigt_to_strain_tensor(stress_voigt)
        stress_tensor_rotated = rotation_matrix.T @ stress_tensor @ rotation_matrix
        stress_voigt_rotated = stress_tensor_to_voigt(stress_tensor_rotated)
        strain_voigt_rotated = S @ stress_voigt_rotated
        strain_tensor_rotated = voigt_to_strain_tensor(strain_voigt_rotated)
        strain_tensor = rotation_matrix @ strain_tensor_rotated @ rotation_matrix.T

    I.e we interpret the rotation_matrix to be the rotation transform that bring a pre-defined cooridnate
    system into the coordinate system of the stress_tensor. C_prime is then the compliance matrix
    described in the strain_tensor local cooridnate system.

    NOTE: it is often faster to compute S_prime = M @ S @ M.T once and apply it to many strain states compared
    to looping over the above alternative 2.

    Args:
        rotation_matrix (:obj:`numpy.ndarray`): Unitary rotaiton matrix, shape=(6,6,N)

    Returns:
        :obj:`numpy.ndarray`: compliance rotation matrix, shape=(6,6,N)
    """
    Uxx, Uxy, Uxz = rotation_matrix[0]
    Uyx, Uyy, Uyz = rotation_matrix[1]
    Uzx, Uzy, Uzz = rotation_matrix[2]

    N = np.array([
        [Uxx**2, Uxy**2, Uxz**2, Uxy*Uxz, Uxx*Uxz, Uxx*Uxy],
        [Uyx**2, Uyy**2, Uyz**2, Uyy*Uyz, Uyx*Uyz, Uyx*Uyy],
        [Uzx**2, Uzy**2, Uzz**2, Uzy*Uzz, Uzx*Uzz, Uzx*Uzy],
        [2*Uyx*Uzx, 2*Uyy*Uzy, 2*Uyz*Uzz, Uyy*Uzz + Uyz*Uzy, Uyx*Uzz + Uyz*Uzx, Uyx*Uzy + Uyy*Uzx],
        [2*Uxx*Uzx, 2*Uxy*Uzy, 2*Uxz*Uzz, Uxy*Uzz + Uxz*Uzy, Uxx*Uzz + Uxz*Uzx, Uxx*Uzy + Uxy*Uzx],
        [2*Uxx*Uyx, 2*Uxy*Uyy, 2*Uxz*Uyz, Uxy*Uyz + Uxz*Uyy, Uxx*Uyz + Uxz*Uyx, Uxx*Uyy + Uxy*Uyx]
    ])
    return N

def rotate_elasticity_matrix(elasticity_matrix, rotation_matrix):
    """Rotate the elasticity matrix.

    Args:
        elasticity_matrix (:obj:`numpy.ndarray`): Positive defenite elasicity matrix, shape=(6,6)
        rotation_matrix (:obj:`numpy.ndarray`): Unitary rotaiton matrix, shape=(6,6,N)

    Returns:
        :obj:`numpy.ndarray`: The rotated elasticity matrix shape=(6,6,N)
    """
    M = get_elasticity_rotation_matrix(rotation_matrix)
    elasticity_matrix_rotated = M @ elasticity_matrix @ M.T
    return elasticity_matrix_rotated


def rotate_compliance_matrix(compliance_matrix, rotation_matrix):
    """Rotate the compliance matrix.

    Args:
        compliance_matrix (:obj:`numpy.ndarray`): Positive defenite compliance matrix, shape=(6,6)
        rotation_matrix (:obj:`numpy.ndarray`): Unitary rotaiton matrix, shape=(6,6,N)

    Returns:
        :obj:`numpy.ndarray`: The rotated compliance matrix shape=(6,6,N)
    """
    N = get_compliance_rotation_matrix(rotation_matrix)
    compliance_matrix_rotated = N @ compliance_matrix @ N.T
    return compliance_matrix_rotated