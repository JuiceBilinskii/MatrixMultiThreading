import numpy as np
import time


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def convert_matrix_to_triangular(matrix: np.ndarray) -> np.ndarray:
    new_matrix = matrix.copy()
    size = len(new_matrix)
    for fd in range(size):
        for i in range(fd + 1, size):
            scale = -new_matrix[i, fd] / new_matrix[fd, fd]
            for j in range(size):
                new_matrix[i, j] = new_matrix[fd, j] * scale + new_matrix[i, j]
    return new_matrix


def calculate_determinant_of_triangular_matrix(triangular_matrix: np.ndarray) -> float:
    size = len(triangular_matrix)
    product = 1.0
    for i in range(size):
        product *= triangular_matrix[i, i]
    return product


matrix_size = 80
left_scope, right_scope = -100.0, 100.0

A = create_squared_matrix(matrix_size, left_scope, right_scope)

start_time = time.time()

AM = convert_matrix_to_triangular(A)
determinant = calculate_determinant_of_triangular_matrix(AM)

end_time = time.time()

print(f'It took {end_time - start_time} seconds')
print(f'Determinant: {determinant}')
# print(f'Numpy determinant: {np.linalg.det(A)}')
