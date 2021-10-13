import numpy as np
import multiprocessing as mp
from functools import reduce


def convert_rows(matrix: np.ndarray, start_row, end_row, focus_diagonal) -> np.ndarray:
    size = len(matrix)
    for i in range(start_row, end_row):
        scale = matrix[i, focus_diagonal] / matrix[focus_diagonal, focus_diagonal]
        for j in range(focus_diagonal, size):
            matrix[i, j] = matrix[focus_diagonal, j] * scale - matrix[i, j]
    return matrix[start_row:end_row]


def convert_matrix_to_triangular_multiprocess(matrix: np.ndarray, number_of_processes=2) -> np.ndarray:
    size = len(matrix)

    for focus_diagonal in range(size - 1):
        step = (size - focus_diagonal - 1) / number_of_processes
        row_scopes = [(round(i * step + focus_diagonal + 1), round((i + 1) * step + focus_diagonal + 1))
                      for i in range(number_of_processes)]

        with mp.Pool(processes=number_of_processes) as pool:
            arguments = ((matrix, start_row, end_row, focus_diagonal) for start_row, end_row in row_scopes)
            matrices = pool.starmap(convert_rows, arguments)

            converted_rows = reduce(lambda first_arr, second_arr: np.concatenate((first_arr, second_arr)), matrices)
            matrix = np.concatenate((matrix[:focus_diagonal + 1], converted_rows))

    return matrix


def convert_matrix_to_triangular_single_process(matrix: np.ndarray) -> np.ndarray:
    size = len(matrix)
    for focus_diagonal in range(size):
        for i in range(focus_diagonal + 1, size):
            scale = matrix[i, focus_diagonal] / matrix[focus_diagonal, focus_diagonal]
            for j in range(focus_diagonal, size):
                matrix[i, j] = matrix[focus_diagonal, j] * scale - matrix[i, j]
    return matrix
