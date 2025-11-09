from sklearn.model_selection import train_test_split
from mlp_classifier_own import MLPClassifierOwn
import numpy as np

def train_nn_own(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifierOwn:
    """
    Train MLPClassifierOwn with PCA-projected features.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifierOwn object
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Create a MLPClassifierOwn object and fit it using (X_train, y_train)
    #       Print the train accuracy and validation accuracy
    #       Return the trained model

    alpha = 0.0
    
    mlp0 = MLPClassifierOwn(num_epochs=5 ,alpha=alpha, hidden_layer_sizes=[16], random_state=42)
    mlp0.fit(X_train, y_train)
    tra_acc = mlp0.score(X_train, y_train)
    val_acc = mlp0.score(X_val, y_val)
    print(f"For alpha = {alpha}:")
    print(f"Train accuracy:\t\t{tra_acc}")
    print(f"Validation accuracy:\t{val_acc}")

    return mlp0