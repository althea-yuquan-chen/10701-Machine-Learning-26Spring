import numpy as np
import copy
import time
import math

# You may use your own arguments and return values depending on your implementation
def load_split_data():
    '''
    Load temperature.csv and split data
    '''
    data = np.genfromtxt('temperature.csv', delimiter=',', skip_header=1, usecols=1)
    # Split data into train and test sets
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

# You may use your own arguments and return values depending on your implementation.
def create_matrices(data, D): 
    '''
    Create data matrices 
    '''
    T = len(data)
    # Number of samples we can create
    num_samples = T - D
    
    # Initialize X and y
    # X shape: (T-D, D)
    # y shape: (T-D,)
    X = np.zeros((num_samples, D))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        X[i, :] = data[i : i + D]
        y[i] = data[i + D]
        
    return X, y
    return 

# (5.2.1, 5.2.2) OLS estimation 
# This function will be autograded, please use the same arguments and return values as specified.
def OLS(D, X_train, y_train, X_test, y_test):
    '''
    Arguments: 
        D: number of timesteps
        X_train: 2D numpy array of size ((T-D), D)
        y_train: 1D numpy array of length (T-D)
        X_test: 2D numpy array of size (C, D)
        y_test: 1D numpy array of length C      
    Returns: 
        ols_weights: List (Weights for first 10 features and the bias weight as the 11th element)
        ols_time: Float (Number of seconds taken to fit OLS)
        ols_test_mse: Float (MSE of model on test data)
    '''
    # Start the timer
    start_time = time.time()
    
    # Add bias column (column of ones) to X_train and X_test
    # We add it as the last column to make indexing consistent
    X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
    # Solve the Normal Equation: w = inv(X^T * X) * X^T * y
    xtx = X_train_bias.T @ X_train_bias
    xty = X_train_bias.T @ y_train
    weights = np.linalg.solve(xtx, xty)
    
    # Stop the timer
    ols_time = time.time() - start_time
    
    # Predict on test set and calculate MSE
    predictions = X_test_bias @ weights
    ols_test_mse = np.mean((y_test - predictions) ** 2)
    
    # Format the returned weights
    # We need the first 10 feature weights and the bias (the very last element)
    first_10_weights = weights[:10].tolist()
    bias_weight = weights[-1]
    
    # If D < 10, the prompt implies we just take what we have. 
    # But based on the instructions, we expect D >= 10 for these questions.
    ols_weights = first_10_weights + [bias_weight]
    
    return ols_weights, ols_time, ols_test_mse


# (5.2.4) SGD estimation
# This function will be autograded, please use the same arguments and return values as specified.
def SGD(D,  X_train, y_train, X_test, y_test, num_epochs = 20, lr = 1e-10):
    '''
    Arguments: 
        D: number of timesteps
        X_train: 2D numpy array of size ((T-D), D)
        y_train: 1D numpy array of length (T-D)
        X_test: 2D numpy array of size (C, D)
        y_test: 1D numpy array of length C      
        num_epochs: Int (number of epochs)
        lr: Float (learning rate)
    Returns: 
        sgd_train_mse: Float (MSE of model on train data)
        sgd_test_mse: Float (MSE of model on test data)
        sgd_time: Float (Number of seconds taken to train SGD)
        sgd_weights: List (Weights for first 10 features and the bias weight as the 11th element)
    '''
    # Initialize weights and bias
    weights = np.full(D, 1.0 / D)
    bias = 1.0
    
    num_train = X_train.shape[0]

    start_time = time.time()
    
    # SGD training loop
    for epoch in range(num_epochs):
        # No shuffling, iterate through the training data in order
        for i in range(num_train):
            xi = X_train[i]
            yi = y_train[i]
            
            # y_hat = w·x + b
            prediction = np.dot(xi, weights) + bias
            
            # Bias: (y_hat - y)
            error = prediction - yi
            
            # Update weights and bias using the gradients
            # Gradient: grad_w = error * x, grad_b = error
            weights = weights - lr * error * xi
            bias = bias - lr * error
            
    sgd_time = time.time() - start_time
    
    # MSE on train and test sets
    train_preds = X_train @ weights + bias
    sgd_train_mse = np.mean((train_preds - y_train) ** 2)
    
    test_preds = X_test @ weights + bias
    sgd_test_mse = np.mean((test_preds - y_test) ** 2)
    
    # Output the first 10 weights and the bias
    sgd_weights = weights[:10].tolist() + [bias]
    
    return sgd_train_mse, sgd_test_mse, sgd_time, sgd_weights


if __name__ == '__main__':
    # Load, split and preprocess the data
    train_data, test_data = load_split_data()

    # 5.2.1 and 5.2.2: OLS experiments with different D values
    print("="*20 + " OLS EXPERIMENTS " + "="*20)
    D_values = [50, 100, 500]
    ols_results = {}

    for D in D_values:
        X_train, y_train = create_matrices(train_data, D)
        X_test, y_test = create_matrices(test_data, D)
        
        weights, fit_time, mse = OLS(D, X_train, y_train, X_test, y_test)
        ols_results[D] = fit_time
        
        print(f"\n[D = {D}]")
        print(f"Time taken to fit: {fit_time:.4f} seconds")
        print(f"Test MSE: {mse:.4f}")
        print(f"Weights (First 10 + Bias): {[round(w, 6) for w in weights]}")
    
    # 5.2.4: SGD experiment with D=17520
    # print("\n" + "="*20 + " SGD EXPERIMENT (D=17520) " + "="*20)
    # D_sgd = 17520
    # X_train_sgd, y_train_sgd = create_matrices(train_data, D_sgd)
    # X_test_sgd, y_test_sgd = create_matrices(test_data, D_sgd)
    
    # train_mse, test_mse, sgd_time, sgd_w = SGD(D_sgd, X_train_sgd, y_train_sgd, 
    #                                            X_test_sgd, y_test_sgd, 
    #                                            num_epochs=20, lr=1e-10)
    
    # print(f"Train MSE: {train_mse:.4f}")
    # print(f"Test MSE: {test_mse:.4f}")
    # print(f"Total training time: {sgd_time:.2f} seconds")
    # print(f"First 10 weights: {[round(w, 8) for w in sgd_w[:10]]}")
    # print(f"Bias weight: {sgd_w[-1]:.8f}")
    


    