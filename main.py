import numpy as np
import time
import threading


def convert_row(matrix, matrix_row, focus_diagonal, size):
    scale = matrix[matrix_row, focus_diagonal] / matrix[focus_diagonal, focus_diagonal]
    for j in range(focus_diagonal, size):
        matrix[matrix_row, j] = matrix[focus_diagonal, j] * scale - matrix[matrix_row, j]


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def convert_matrix_to_triangular(matrix: np.ndarray) -> np.ndarray:
    size = len(matrix)
    for focus_diagonal in range(size):
        threads = []
        for i in range(focus_diagonal + 1, size):
            threads.append(threading.Thread(target=convert_row, args=(matrix, i, focus_diagonal, size)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return matrix


def calculate_determinant_of_triangular_matrix(triangular_matrix: np.ndarray) -> float:
    size = len(triangular_matrix)
    product = 1.0
    for i in range(size):
        product *= triangular_matrix[i, i]
    return product


if __name__ == '__main__':
    matrix_size = 140
    left_scope, right_scope = -1.0, 1.0

    squared_matrix = create_squared_matrix(matrix_size, left_scope, right_scope)
    matrix_copy = squared_matrix.copy()

    start_time = time.time()

    triangular_matrix = convert_matrix_to_triangular(squared_matrix)
    determinant = calculate_determinant_of_triangular_matrix(triangular_matrix)

    end_time = time.time()

    print(f'It took {end_time - start_time} seconds')

    print(f'Determinant: {determinant}')
    print(f'Numpy determinant: {np.linalg.det(matrix_copy)}')
