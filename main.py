import time
from determinant_methods import gaussian_method, minor_method, numpy_method, utils


def test_minor_method():
    matrix_size = 10
    left_scope, right_scope = -1.0, 1.0

    initial_matrix = utils.create_squared_matrix(matrix_size, left_scope, right_scope)

    start_time = time.time()
    determinant = minor_method.calculate_matrix_determinant_multiprocess(initial_matrix, 8)
    end_time = time.time()

    print(f'Multiprocess execution time: {end_time - start_time}')
    print(f'Multiprocess determinant: {determinant}\n')

    start_time = time.time()
    determinant = minor_method.calculate_matrix_determinant_single_process(initial_matrix,
                                                                           start_column=0,
                                                                           end_column=len(initial_matrix))
    end_time = time.time()

    print(f'Single process execution time: {end_time - start_time}')
    print(f'Single process determinant: {determinant}\n')

    print(f'Numpy determinant {numpy_method.calculate_numpy_matrix_determinant(initial_matrix)}')


def test_gaussian_method():
    matrix_size = 700
    left_scope, right_scope = -1.0, 1.0

    initial_matrix = utils.create_squared_matrix(matrix_size, left_scope, right_scope)
    matrix_a = initial_matrix.copy()
    matrix_b = initial_matrix.copy()

    start_time = time.time()
    converted_matrix = gaussian_method.convert_matrix_to_triangular_multiprocess(matrix_a, 2)
    determinant = utils.calculate_triangular_matrix_determinant(converted_matrix)
    end_time = time.time()

    print(f'Multiprocess execution time: {end_time - start_time} seconds')
    print(f'Determinant: {determinant}\n')

    start_time = time.time()
    gaussian_method.convert_matrix_to_triangular_single_process(matrix_b)
    determinant = utils.calculate_triangular_matrix_determinant(matrix_b)
    end_time = time.time()
    print(f'Single process execution time: {end_time - start_time} seconds')
    print(f'Determinant: {determinant}\n')

    print(f'Numpy determinant: {numpy_method.calculate_numpy_matrix_determinant(initial_matrix)}')


if __name__ == '__main__':
    test_gaussian_method()
    # test_minor_method()
