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
