import numpy as np
import time


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def convert_matrix_to_triangular(matrix: np.ndarray) -> np.ndarray:
    size = len(matrix)
    for focus_diagonal in range(size):
        for i in range(focus_diagonal + 1, size):
            scale = matrix[i, focus_diagonal] / matrix[focus_diagonal, focus_diagonal]
            for j in range(focus_diagonal, size):
                matrix[i, j] = matrix[focus_diagonal, j] * scale - matrix[i, j]
    return matrix


def calculate_determinant_of_triangular_matrix(triangular_matrix: np.ndarray) -> float:
    size = len(triangular_matrix)
    product = 1.0
    for i in range(size):
        product *= triangular_matrix[i, i]
    return product


matrix_size = 350
left_scope, right_scope = -1.0, 1.0

A = create_squared_matrix(matrix_size, left_scope, right_scope)

start_time = time.time()

AM = convert_matrix_to_triangular(A)
determinant = calculate_determinant_of_triangular_matrix(AM)

end_time = time.time()

print(f'Matrix size: {matrix_size}')
print(A)
print(f'Execution time: {end_time - start_time} seconds')

print(f'Determinant: {determinant}')
print(f'Numpy determinant: {np.linalg.det(A)}')
