import numpy as np
import time
import threading


def convert_rows(matrix, fd, size, start_row, end_row):
    for i in range(start_row, end_row):
        scale = matrix[i, fd] / matrix[fd, fd]
        for j in range(fd, size):
            matrix[i, j] = matrix[fd, j] * scale - matrix[i, j]


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def thread_convert_matrix_to_triangular(matrix: np.ndarray) -> None:
    size = len(matrix)
    for fd in range(size):
        start_row = fd + 1
        middle_row = (start_row + size) // 2

        if start_row != middle_row:
            thread_a = threading.Thread(target=convert_rows, args=(matrix, fd, size, start_row, middle_row))
            thread_b = threading.Thread(target=convert_rows, args=(matrix, fd, size, middle_row, size))

            thread_a.start()
            thread_b.start()

            thread_a.join()
            thread_b.join()
        else:
            thread = threading.Thread(target=convert_rows, args=(matrix, fd, size, start_row, size))
            thread.start()
            thread.join()


def non_thread_convert_matrix_to_triangular(matrix: np.ndarray) -> None:
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
    matrix_size = 150
    left_scope, right_scope = -1.0, 1.0

    squared_matrix_a = create_squared_matrix(matrix_size, left_scope, right_scope)
    squared_matrix_b = squared_matrix_a.copy()
    matrix_copy = squared_matrix_a.copy()

    start_time = time.time()
    thread_convert_matrix_to_triangular(squared_matrix_a)
    end_time = time.time()

    print(f'Thread: {end_time - start_time} seconds')

    start_time = time.time()
    non_thread_convert_matrix_to_triangular(squared_matrix_b)
    end_time = time.time()

    print(f'Non-thread: {end_time - start_time} seconds')

    determinant = calculate_determinant_of_triangular_matrix(squared_matrix_b)

    print(f'Determinant: {determinant}')
    print(f'Numpy determinant: {np.linalg.det(matrix_copy)}')
