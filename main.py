import numpy as np
import time
import threading


def convert_row(matrix, matrix_row, focus_diagonal, size):
    scale = matrix[matrix_row, focus_diagonal] / matrix[focus_diagonal, focus_diagonal]
    for j in range(focus_diagonal, size):
        matrix[matrix_row, j] = matrix[focus_diagonal, j] * scale - matrix[matrix_row, j]


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def convert_matrix_to_triangular(matrix: np.ndarray) -> None:
    size = len(matrix)
    for focus_diagonal in range(size):
        threads = []
        for i in range(focus_diagonal + 1, size):
            threads.append(threading.Thread(target=convert_row, args=(matrix, i, focus_diagonal, size)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


def iteration_convert_matrix_to_triangular(matrix: np.ndarray) -> None:
    size = len(matrix)
    for fd in range(size):
        for i in range(fd + 1, size):
            scale = matrix[i, fd] / matrix[fd, fd]
            for j in range(fd, size):
                matrix[i, j] = matrix[fd, j] * scale - matrix[i, j]


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
    convert_matrix_to_triangular(squared_matrix)
    end_time = time.time()

    print(f'It took {end_time - start_time} seconds')

    start_time = time.time()
    iteration_convert_matrix_to_triangular(matrix_copy)
    end_time = time.time()

    print(f'It took {end_time - start_time} seconds')

    determinant = calculate_determinant_of_triangular_matrix(squared_matrix)

    print(f'Determinant: {determinant}')
    print(f'Numpy determinant: {np.linalg.det(matrix_copy)}')
