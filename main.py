import numpy as np
import time
import multiprocessing


def create_squared_matrix(size: int, low: float, high: float) -> np.ndarray:
    return np.random.uniform(low, high, size=(size, size))


def calculate_recursive_determinant(matrix: np.ndarray) -> float:
    if len(matrix) == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]

    total = 0
    for focus_column in range(len(matrix)):
        copied_matrix = np.delete(matrix[1:], focus_column, 1)
        sign = (-1) ** (focus_column % 2)
        sub_determinant = calculate_recursive_determinant(copied_matrix)
        total += sign * matrix[0, focus_column] * sub_determinant
    return total


def multiprocess_calculate_recursive_determinant(matrix: np.ndarray, start_column, end_column) -> float:
    if len(matrix) == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]

    total = 0
    for focus_column in range(start_column, end_column):
        copied_matrix = np.delete(matrix[1:], focus_column, 1)
        sign = (-1) ** (focus_column % 2)
        sub_determinant = calculate_recursive_determinant(copied_matrix)
        total += sign * matrix[0, focus_column] * sub_determinant
    print(total)
    # return total


def manage_multiprocess_determinant_calculation(matrix):
    process_a = multiprocessing.Process(target=multiprocess_calculate_recursive_determinant,
                                        args=(matrix, 0, len(matrix) // 2))
    process_b = multiprocessing.Process(target=multiprocess_calculate_recursive_determinant,
                                        args=(matrix, len(matrix) // 2, len(matrix)))

    process_a.start()
    process_b.start()

    process_a.join()
    process_b.join()


if __name__ == '__main__':
    matrix_size = 10
    left_scope, right_scope = -1.0, 1.0

    initial_matrix = create_squared_matrix(matrix_size, left_scope, right_scope)

    start_time = time.time()
    manage_multiprocess_determinant_calculation(initial_matrix)
    end_time = time.time()
    print(f'Recursive multiprocess determinant: {end_time - start_time}')

    start_time = time.time()
    determinant = calculate_recursive_determinant(initial_matrix)
    end_time = time.time()
    print(f'Recursive single process determinant: {end_time - start_time}')
    print(determinant)

    print(np.linalg.det(initial_matrix))
