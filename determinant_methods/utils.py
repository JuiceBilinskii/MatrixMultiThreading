import numpy as np
from functools import reduce


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def calculate_triangular_matrix_determinant(triangular_matrix: np.ndarray) -> float:
    size = len(triangular_matrix)
    product = reduce(lambda a, b: a * b, (triangular_matrix[i, i] for i in range(size)))
    return product
