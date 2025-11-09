import numpy as np



def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)
    squared_error = ((theta[1] * x + theta[0]) - y)**2
    return np.mean(squared_error)


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.1.1)
    w_numerator = N * np.sum(x * y) - np.sum(x) * np.sum(y)
    w_denominator = N * np.sum(x**2) - (np.sum(x))**2

    w = w_numerator/w_denominator
    b = np.mean(y) - w * np.mean(x)
    return np.array([b, w])


    


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    # TODO: Implement Pearson correlation coefficient, as shown in Equation 3 (Task 1.1.2).
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    covariance_sum = np.sum((x - x_mean) * (y - y_mean))
    standard_deviations_product = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    pearson_r = covariance_sum/standard_deviations_product
    return pearson_r




def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    design_matrix = np.insert(data, 0, 1, axis=1)
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)
    squared_error = ((X @ theta) - y)**2
    return np.mean(squared_error)


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.2.1). 
    # Note: Use the pinv function.
    theta = pinv(X) @ y
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)
    x_column = x.reshape(-1, 1)
    polynomial_design_matrix = compute_design_matrix(x)
    for i in range(2, K + 1):
        polynomial_design_matrix = np.hstack((polynomial_design_matrix, x_column**i))
    
    return polynomial_design_matrix