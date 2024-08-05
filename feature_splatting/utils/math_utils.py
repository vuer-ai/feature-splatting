import numpy as np

def point_to_plane_distance(point, plane):
    """
    Compute distance from point to plane
    :param point: point (x, y, z)
    :param plane: plane (A, B, C, D)
    :return: distance from point to plane
    """
    x, y, z = point
    A, B, C, D = plane
    numerator = np.abs(A*x + B*y + C*z + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    distance = numerator / denominator
    return distance

def vector_angle(vec_a, vec_b):
    """
    Calculate angle between two vectors
    :param vec_a: vector a
    :param vec_b: vector b
    :return: angle between two vectors
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    cos_theta = dot / (norm_a * norm_b)
    theta = np.arccos(cos_theta)
    return theta
