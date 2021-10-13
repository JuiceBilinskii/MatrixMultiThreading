import numpy as np
import multiprocessing as mp
import time


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def calculate_matrix_determinant(matrix: np.ndarray, start_column, end_column) -> float:
    if len(matrix) == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]

    total = 0
    for focus_column in range(start_column, end_column):
        minor_matrix = np.delete(matrix[1:], focus_column, 1)
        sign = (-1) ** (focus_column % 2)
        sub_determinant = calculate_matrix_determinant(minor_matrix, start_column=0, end_column=len(minor_matrix))
        total += sign * matrix[0, focus_column] * sub_determinant
    return total


def manage_multiprocess_determinant_calculation(matrix, number_of_processes=2):
    step = len(matrix) / number_of_processes
    column_scopes = [(round(i * step), round((i + 1) * step)) for i in range(number_of_processes)]

    with mp.Pool(processes=number_of_processes) as pool:
        arguments = ((matrix, start_column, end_column) for start_column, end_column in column_scopes)
        sub_determinants = pool.starmap(calculate_matrix_determinant, arguments)
        return sum(sub_determinants)


if __name__ == '__main__':
    matrix_size = 10
    left_scope, right_scope = -1.0, 1.0

    initial_matrix = create_squared_matrix(matrix_size, left_scope, right_scope)

    start_time = time.time()
    determinant = manage_multiprocess_determinant_calculation(initial_matrix, 2)
    end_time = time.time()

    print(f'Multiprocess determinant calculation execution time: {end_time - start_time}')
    print(f'Multiprocess determinant: {determinant}\n')

    start_time = time.time()
    determinant = calculate_matrix_determinant(initial_matrix, start_column=0, end_column=len(initial_matrix))
    end_time = time.time()

    print(f'Single process determinant calculation execution time: {end_time - start_time}')
    print(f'Single process determinant: {determinant}\n')

    print(f'Numpy determinant {np.linalg.det(initial_matrix)}')
