from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings

# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.

    pca = PCA(n_components=n_components, random_state=42)
    X_hat = pca.fit_transform(X_train) 
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance}")
    return X_hat, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons and hidden layers.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of layers/neurons.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.,
    hidden_layers = [(2, ), (8, ), (64, ), (256, ), (1024, ), (128, 256, 128)]
    best_classifier = None
    best_accuracy_v = 0
   
    for layer in hidden_layers:
        classifier = MLPClassifier(hidden_layer_sizes=layer, max_iter=100, solver="adam", random_state=1)
        classifier.fit(X_train, y_train)
        

        accuracy_t = classifier.score(X_train, y_train)
        accuracy_v = classifier.score(X_val, y_val)
        final_loss = classifier.loss_

        print(f"Hidden layer: {layer}")
        print(f"Train accuracy: {accuracy_t}")
        print(f"Validation accuracy: {accuracy_v}")
        print(f"Final loss: {final_loss}")

        if accuracy_v > best_accuracy_v:
            best_accuracy_v=accuracy_v
            best_classifier=classifier    

    return best_classifier


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.

    hidden_layers = [(2, ), (8, ), (64, ), (256, ), (1024, ), (128, 256, 128)]
    best_classifier = None
    best_accuracy_v = 0
   
    for layer in hidden_layers:
        # classifier = MLPClassifier(hidden_layer_sizes=layer, max_iter=100, solver="adam", random_state=1, early_stopping=True, alpha=0.1)
        # classifier = MLPClassifier(hidden_layer_sizes=layer, max_iter=100, solver="adam", random_state=1, early_stopping=True)
        classifier = MLPClassifier(hidden_layer_sizes=layer, max_iter=100, solver="adam", random_state=1, alpha=0.1)
        classifier.fit(X_train, y_train)
        

        accuracy_t = classifier.score(X_train, y_train)
        accuracy_v = classifier.score(X_val, y_val)
        final_loss = classifier.loss_

        print(f"Hidden layer: {layer}")
        print(f"Train accuracy: {accuracy_t}")
        print(f"Validation accuracy: {accuracy_v}")
        print(f"Final loss: {final_loss}")

        if accuracy_v > best_accuracy_v:
            best_accuracy_v=accuracy_v
            best_classifier=classifier    

    return best_classifier

    


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.
    plt.plot(nn.loss_curve_, label="Training loss")
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def custom_accuracy_score(y_true, y_pred):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / len(y_true)


def custom_precision_score(y_true, y_pred):
    labels = list(set(y_true))
    total = len(y_true)
    weighted_precision = 0

    for label in labels:
        tp = 0
        fp =0
        count = 0

        for yt, yp in zip(y_true, y_pred):
            if yp == label:
                if yt == yp:
                    tp += 1
                else:
                    fp += 1
            if yt == label:
                count  += 1


        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        weight = count / total
        weighted_precision += weight * precision

    return weighted_precision


def custom_recall_score(y_true, y_pred):
    labels = list(set(y_true))
    total = len(y_true)
    weighted_recall = 0

    for label in labels:
        tp = 0
        fn = 0
        count = 0

        for yt, yp in zip(y_true, y_pred):
            if yt == label:
                count += 1
                if yp == label:
                    tp += 1
                else:
                    fn += 1

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        weight = count / total
        weighted_recall += weight * recall

    return weighted_recall


def custom_f1_score(y_true, y_pred):
    labels = list(set(y_true))
    total = len(y_true)
    weighted_f1 = 0

    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        count = 0

        for yt, yp in zip(y_true, y_pred):
            if yt == label:
                count += 1
                if yp == label:
                    tp += 1
                else:
                    fn += 1
            elif yp == label:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        weight = count / total
        weighted_f1 += weight * f1

    return weighted_f1





def find_best_model(X_train: np.ndarray, y_train: np.ndarray, model_reg: MLPClassifier, model_gs: MLPClassifier) -> MLPClassifier:
     
     best_model = None
     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)
     
     y_pred_reg = model_reg.predict(X_val)
     y_pred_gs = model_gs.predict(X_val)

     accuracy_reg = custom_accuracy_score(y_val, y_pred_reg)
     precision_reg = custom_precision_score(y_val, y_pred_reg)
     recall_reg = custom_recall_score(y_val, y_pred_reg)
     f1_reg = custom_f1_score(y_val, y_pred_reg)

     accuracy_gs = custom_accuracy_score(y_val, y_pred_gs)
     precision_gs = custom_precision_score(y_val, y_pred_gs)
     recall_gs = custom_recall_score(y_val, y_pred_gs)
     f1_gs = custom_f1_score(y_val, y_pred_gs)

     
     metrics_reg = {
         'accuracy': accuracy_reg,
         'precision': precision_reg,
         'recall': recall_reg,
         'f1_score': f1_reg
     }

     metrics_gs = {
         'accuracy': accuracy_gs,
         'precision': precision_gs,
         'recall': recall_gs,
         'f1_score': f1_gs
     }

     reg_better = sum(metrics_reg[i] > metrics_gs[i] for i in metrics_reg)
     gs_better = sum(metrics_gs[i] > metrics_reg[i] for i in metrics_gs)

     if reg_better > gs_better:
         best_model = model_reg
     elif gs_better > reg_better:
         best_model = model_gs
     else:
         best_model = model_reg if metrics_reg['recall'] >= metrics_gs['recall'] else model_gs

     return best_model
    

def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.
    y_pred = nn.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    for_display = ConfusionMatrixDisplay(matrix)
    for_display.plot()
    plt.title("Confusion Matrix")
    plt.show()

    print(f"Classification report: ")
    print(f"{classification_report(y_test, y_pred)}")


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    params = {
        "alpha": [0.0, 0.1, 1.0],
        "batch_size": [32, 512],
        "hidden_layer_sizes": [(128, ), (256, )]
    }
    classifier = MLPClassifier(max_iter=100, solver='adam', random_state=42)
    grid_search = GridSearchCV(classifier, params, cv=5, verbose=4)
    grid_search.fit(X_train, y_train)

    print(f"Best score: {grid_search.best_score_}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_