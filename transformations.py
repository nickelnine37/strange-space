import numpy as np

def rotate_x(theta_x: float) -> np.ndarray:
    """
    Rotation matrix: rotate theta_x radians clockwise about the x-axis

    Parameters
    ----------
    theta_x     Angle in radians

    Returns
    -------
    Rx          3x3 X rotation matrix
    """
    cos = np.cos(theta_x)
    sin = np.sin(theta_x)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])


def rotate_y(theta_y: float) -> np.ndarray:
    """
    Rotation matrix: rotate theta_y radians clockwise about the y-axis

    Parameters
    ----------
    theta_y     Angle in radians

    Returns
    -------
    Ry          3x3 Y rotation matrix
    """
    cos = np.cos(theta_y)
    sin = np.sin(theta_y)
    return np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])


def rotate_z(theta_z: float) -> np.ndarray:
    """
    Rotation matrix: rotate theta_z radians clockwise about the z-axis

    Parameters
    ----------
    theta_z     Angle in radians

    Returns
    -------
    Rz          3x3 Z rotation matrix
    """
    cos = np.cos(theta_z)
    sin = np.sin(theta_z)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def rotate(theta_x: float, theta_y: float, theta_z: float):
    return rotate_x(theta_x) @ rotate_y(theta_y) @ rotate_z(theta_z)


def norm(v: np.ndarray):
    return v / (v ** 2).sum() ** 0.5