import numpy as np


def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 1

    xIsGreaterEqual10 =  (X_data[:,0] >= 10).astype(int).reshape(-1, 1)
    yIsGreater20 = (X_data[:,1] > 20).astype(int).reshape(-1, 1)

    x = X_data.copy()[:,0].reshape(-1, 1)
    y = X_data.copy()[:,1].reshape(-1, 1)

    inSquare = ((x >= 10) & (y <= 20)).astype(int).reshape(-1, 1)

    X = np.hstack((X_data, inSquare))

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2
    x = X_data.copy()[:,0].reshape(-1, 1)
    y = X_data.copy()[:,1].reshape(-1, 1)
    X = np.hstack((X_data.copy(), (x**2 + y**2 <= 24.5**2).astype(int)))
    #X = np.hstack((X_data.copy(), x**2 + y**2))

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    # X = np.cos(X_data.copy())
    onlyX = X_data.copy()[:,0].reshape(-1,1)
    #X = np.hstack((X_data, onlyX**2, onlyX**3, onlyX**4, onlyX**5, onlyX**6, onlyX**7, onlyX**8, onlyX**9))
    #X = np.hstack((X_data, onlyX**2, onlyX**3, onlyX**4, onlyX**5, onlyX**6, onlyX**7))
    X = np.hstack((X_data, onlyX**2, onlyX**3, onlyX**4, onlyX**5))

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': None, 'solver': 'saga', 'max_iter': 10000, 'tol': 0.000001}

# for 10000 iterations and tol=0.000001
    #l1=0.0: 94.72%
    #l1=1.0: 95.25%
    #none: 95.77

