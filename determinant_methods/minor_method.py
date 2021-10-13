import numpy as np
import multiprocessing as mp


def calculate_matrix_determinant_multiprocess(matrix, number_of_processes=2):
    step = len(matrix) / number_of_processes
    column_scopes = [(round(i * step), round((i + 1) * step)) for i in range(number_of_processes)]

    with mp.Pool(processes=number_of_processes) as pool:
        arguments = ((matrix, start_column, end_column) for start_column, end_column in column_scopes)
        sub_determinants = pool.starmap(calculate_matrix_determinant_single_process, arguments)
        return sum(sub_determinants)


def calculate_matrix_determinant_single_process(matrix: np.ndarray, start_column, end_column) -> float:
    if len(matrix) == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]

    determinant = 0
    for focus_column in range(start_column, end_column):
        minor_matrix = np.delete(matrix[1:], focus_column, 1)
        sign = (-1) ** (focus_column % 2)
        sub_determinant = calculate_matrix_determinant_single_process(minor_matrix,
                                                                      start_column=0, end_column=len(minor_matrix))
        determinant += sign * matrix[0, focus_column] * sub_determinant
    return determinant
