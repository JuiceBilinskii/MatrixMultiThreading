import numpy as np
import time
import threading


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def convert_rows(matrix, fd, size, start_row, end_row):
    for i in range(start_row, end_row):
        scale = matrix[i, fd] / matrix[fd, fd]
        for j in range(fd, size):
            matrix[i, j] = matrix[fd, j] * scale - matrix[i, j]


def thread_convert_matrix_to_triangular(matrix: np.ndarray) -> None:
    size = len(matrix)
    for fd in range(size):
        start_row = fd + 1
        middle_row = (start_row + size) // 2

        if fd < 100:
            thread_a = threading.Thread(target=convert_rows, args=(matrix, fd, size, start_row, middle_row))
            thread_b = threading.Thread(target=convert_rows, args=(matrix, fd, size, middle_row, size))

            thread_a.start()
            thread_b.start()

            thread_a.join()
            thread_b.join()
        else:
            for i in range(fd + 1, size):
                scale = matrix[i, fd] / matrix[fd, fd]
                for j in range(fd, size):
                    matrix[i, j] = matrix[fd, j] * scale - matrix[i, j]


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
    matrix_size = 200
    left_scope, right_scope = -1.0, 1.0

    initial_matrix = create_squared_matrix(matrix_size, left_scope, right_scope)
    matrix_a = initial_matrix.copy()
    matrix_b = initial_matrix.copy()

    start_time = time.time()
    thread_convert_matrix_to_triangular(matrix_a)
    end_time = time.time()
    print(f'Thread: {end_time - start_time} seconds')

    start_time = time.time()
    non_thread_convert_matrix_to_triangular(matrix_b)
    end_time = time.time()
    print(f'Non-thread: {end_time - start_time} seconds')

    print(f'Thread Determinant: {calculate_determinant_of_triangular_matrix(matrix_a)}')
    print(f'Non-thread Determinant: {calculate_determinant_of_triangular_matrix(matrix_b)}')
    print(f'Numpy determinant: {np.linalg.det(initial_matrix)}')
