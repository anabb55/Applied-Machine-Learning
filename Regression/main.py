import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import rastrigin, gradient_rastrigin, gradient_descent, finite_difference_gradient_approx


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, 
                    "duration": 4, "exercise_intensity": 5, 
                    "fitness_level": 6, "calories": 7}

    # After loading the data, you can for example access it like this: 
    # `smartwatch_data[:, column_to_id['hours_sleep']]`
    smartwatch_data = np.load('data/smartwatch_data.npy')
    exc_duration = smartwatch_data[:, column_to_id['duration']]
    fitness_level = smartwatch_data[:, column_to_id['fitness_level']]
    calories_burned = smartwatch_data[:, column_to_id['calories']]
    hours_sleep = smartwatch_data[:, column_to_id['hours_sleep']]
    avg_pulse = smartwatch_data[:, column_to_id['avg_pulse']]
    hours_work = smartwatch_data[:, column_to_id['hours_work']]
    exc_intensity = smartwatch_data[:, column_to_id['exercise_intensity']]
    


    # TODO: Implement Task 1.1.2: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    print(f'Correlation between exc_duration and fitness_level: {calculate_pearson_correlation(exc_duration, fitness_level)}')
    print(f'Correlation between exc_duration and calories_burned: {calculate_pearson_correlation(exc_duration, calories_burned)}')
    print(f'Correlation between exc_duration and exc_intensity: {calculate_pearson_correlation(exc_duration, exc_intensity)}')

    design_matrix_1 = compute_design_matrix(exc_duration)
    
    
    if(use_linalg_formulation == True):
         theta1= fit_multiple_lin_model(design_matrix_1, fitness_level)
         theta2 = fit_multiple_lin_model(design_matrix_1, calories_burned)
         theta3 = fit_multiple_lin_model(design_matrix_1, exc_intensity)

    else:
        theta1 = fit_univariate_lin_model(exc_duration, fitness_level)
        theta2 = fit_univariate_lin_model(exc_duration, calories_burned)
        theta3= fit_univariate_lin_model(exc_duration, exc_intensity)
        

    
    print(f'Parameter vector theta1: {theta1}')
    print(f'Parameter vector theta2: {theta2}')
    print(f'Parameter vector theta3: {theta3}')

    print(f'MSE1: {univariate_loss(exc_duration, fitness_level, theta1)}')
    print(f'MSE2: {univariate_loss(exc_duration, calories_burned, theta2)}')
    print(f'MSE3: {univariate_loss(exc_duration, exc_intensity, theta3)}')
    

    plot_scatterplot_and_line(exc_duration, fitness_level, theta1, 'exercise duration', 'fitness level', 'Linear Regression - Exercise Duration and Fitness Level', 'linreg_exc_duration_fitness_level')
    plot_scatterplot_and_line(exc_duration, calories_burned, theta2, 'exercise duration', 'calories burned', 'Linear Regression - Exercise Duration and Calories Burned', 'linreg_exc_duration_calories_burned' )
    plot_scatterplot_and_line(exc_duration, exc_intensity, theta3, 'exercise duration', 'exercise intensity', 'Linear Regression - Exercise Duration and Exercise Intensity', 'linreg_exc_duration_exc_intensity' )

    print(f'Non-linear data: ')
    
    print(f'Correlation between hours_sleep and avg_pulse: {calculate_pearson_correlation(hours_sleep, avg_pulse)}')
    print(f'Correlation between hours_sleep and exc_duration: {calculate_pearson_correlation(hours_sleep, exc_duration)}')
    print(f'Correlation between hours_work and calories_burned: {calculate_pearson_correlation(hours_work, calories_burned)}')

    design_matrix_4 = compute_design_matrix(hours_sleep)
    design_matrix_5 = compute_design_matrix(hours_work)
    

    if(use_linalg_formulation == True):
         
         theta4 = fit_multiple_lin_model(design_matrix_4, avg_pulse)
         theta5 = fit_multiple_lin_model(design_matrix_4, exc_duration)
         theta6 = fit_multiple_lin_model(design_matrix_5, calories_burned)

    else:
        theta4 = fit_univariate_lin_model(hours_sleep, avg_pulse)
        theta5 = fit_univariate_lin_model(hours_sleep, exc_duration)
        theta6= fit_univariate_lin_model(hours_work, calories_burned)
      

    print(f'Parameter vector theta5: {theta4}')
    print(f'Parameter vector theta5: {theta5}')
    print(f'Parameter vector theta6: {theta6}')

    print(f'MSE4: {univariate_loss(hours_sleep, avg_pulse, theta4)}')
    print(f'MSE5: {univariate_loss(hours_sleep, exc_duration, theta5)}')
    print(f'MSE6: {univariate_loss(hours_work, calories_burned, theta6)}')

    plot_scatterplot_and_line(hours_sleep, avg_pulse, theta4, 'hours sleep', 'average pulse', 'Linear Regression - Hours Sleep and Average Pulse', 'linreg_hours_sleep_avg_pulse' )
    plot_scatterplot_and_line(hours_sleep, exc_duration, theta5, 'hours sleep', 'exercise duration', 'Linear Regression - Hours Sleep and Exercise Duration', 'linreg_hours_sleep_exc_duration' )
    plot_scatterplot_and_line(hours_work, calories_burned, theta6, 'hours work', 'calories burned', 'Linear Regression - Hours Work and Calories Burned', 'linreg_hours_work_calories_burned' )


    pass


    # TODO: Implement Task 1.2.3: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.
    smartwatch_data_selected = smartwatch_data[:, [column_to_id['fitness_level'], column_to_id['exercise_intensity'], column_to_id['duration']]]
    design_matrix = compute_design_matrix(smartwatch_data_selected)
    theta = fit_multiple_lin_model(design_matrix, calories_burned)
    print(f'Parameter vector for multiple linear regression: {theta}')
    MSE = multiple_loss(design_matrix, calories_burned, theta)
    print(f'MSE - multiple lin regression: {MSE}')
    pass


    # TODO: Implement Task 1.3.2: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    design_m = compute_polynomial_design_matrix(exc_duration, 4)
    theta = fit_multiple_lin_model(design_m, exc_intensity)
    print(f'Parameter vector for multiple linear regression: {theta}')
    plot_scatterplot_and_polynomial(exc_duration, exc_intensity, theta, "Exercise Duration", "Exercise Intensity", "Polynomial Linear Regression", "polynomial_regression")
    MSE = multiple_loss(design_m, exc_intensity, theta)
    print(f'MSE - polynomial linear regression: {MSE}')
    
    pass


    # TODO: Implement Task 1.3.3: Use x_small and y_small to fit a polynomial model.
    # Find and report the smallest K that gets zero loss. Plot the data points and the polynomial function.
    x_small = smartwatch_data[:5, column_to_id['duration']]
    y_small = smartwatch_data[:5, column_to_id['calories']]

    design_m = compute_polynomial_design_matrix(x_small, 4)
    theta = fit_multiple_lin_model(design_m, y_small)
    plot_scatterplot_and_polynomial(x_small, y_small, theta, "x", "y", "Polynomial with 5 Data Points", "polynomial_regression_5_points")
    MSE = multiple_loss(design_m, y_small, theta)
    print(MSE)


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # TODO: Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-1.npy')
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # TODO: Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-2.npy')
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-2-data.npy')
            y = np.load('data/targets-dataset-3.npy')
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')



        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')



        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)

        # Fit the model
        clf.fit(X_train, y_train)

        # calculate accuracy
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test) 
        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        yhat_train = clf.predict_proba(X_train)
        yhat_test = clf.predict_proba(X_test)





        # Use the `log_loss` function to calculate the cross-entropy loss
        loss_train, loss_test = log_loss(y_train, yhat_train[:,1]), log_loss(y_test, yhat_test[:,1])
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')



        # Print theta vector
        classifier_weights, classifier_bias = clf.intercept_, clf.coef_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(46)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = np.random.standard_normal()
    y0 = np.random.standard_normal()
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(rastrigin)
        plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0))

    # TODO: Check if gradient_rastrigin is correct at (x0, y0). 
    # To do this, print the true gradient and the numerical approximation.
    print("Finite diff aprox: " + str(finite_difference_gradient_approx(rastrigin, x=x0, y=y0)))
    print("Numerical: " + str(gradient_rastrigin(x=x0, y=y0)))



    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    #x_list, y_list, f_list = gradient_descent(rastrigin, gradient_rastrigin, x0, y0, 0.013, 0.98, 65)
    x_list, y_list, f_list = gradient_descent(rastrigin, gradient_rastrigin, x0, y0, 0.0131, 0.97895, 62)
    #x_list, y_list, f_list = gradient_descent(rastrigin, gradient_rastrigin, x0, y0, 0.011175, 1.0, 1000)


    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {rastrigin(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)


    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    plot_function_over_iterations(f_list=f_list)


def main():
    np.random.seed(46)

    task_1(use_linalg_formulation=True)
    task_2()
    task_3(initial_plot=True)


if __name__ == '__main__':
    main()
