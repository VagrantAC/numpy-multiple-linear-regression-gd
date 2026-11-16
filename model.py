"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    a = n_train
    b = n_train + n_val
    return (
        X[:a], y[:a],
        X[a:b], y[a:b],
        X[b:], y[b:]
    )

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    return np.column_stack([np.ones(X.shape[0]), X])

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X_std = standardize_features(X, mean, std)
    return add_bias_column(X_std)

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    return X @ weights

# Step 8 - mse_loss
import math

def mse_loss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    n = X.shape[0]
    return 2 / n * X.T @ (y_pred - y_true)

# Step 10 - normal_equation
def normal_equation(X, y):
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    np.random.seed(seed)
    return np.random.normal(0, 0.01, n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    pred_y = predict_linear(X, weights)
    mse = mse_gradient(X, y, pred_y)
    return weights - lr * mse

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

