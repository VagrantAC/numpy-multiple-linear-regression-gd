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
def normal_equation(X_design, y):
    w, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return w

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

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    y_train_pred = predict_linear(X_train, weights)
    y_val_pred = predict_linear(X_val, weights)
    return mse_loss(y_train, y_train_pred), mse_loss(y_val, y_val_pred)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    if val_loss < best_val_loss:
        return (val_loss, 0, weights, False)
    return (best_val_loss, wait + 1, best_weights, wait + 1 >= patience)

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features, seed=seed)
    return {
        'weights': weights,
        'best_weights': weights.copy(),
        'best_val_loss': np.inf,
        'wait': 0,
        'train_losses': [],
        'val_losses': [],
        'stopped': False
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    new_weights = gd_step(X_train, y_train, state['weights'], lr)
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, new_weights)

    state['weights'] = new_weights
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)

    b, w, bw, s = update_early_stop_state(
        val_loss, state['best_val_loss'], state['wait'],
        new_weights, state['best_weights'], patience
    )
    state['best_val_loss'] = b
    state['wait'] = w
    state['best_weights'] = bw
    state['stopped'] = s
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    n_features = X_train.shape[1]
    state = init_training_state(n_features, seed=seed)
    for _ in range(epochs):
        run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state['stopped']:
            break
    return state['best_weights'], state['train_losses'], state['val_losses']

# Step 18 - mean_absolute_error
import math

def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / len(y_pred)

# Step 19 - root_mean_squared_error
import math

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(np.sum((y_pred - y_true) ** 2) / len(y_pred))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    ss_res = np.sum((y_pred - y_true) ** 2)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    n = len(train_losses)
    epochs = list(range(1, n + 1))
    train_list = [float(x) for x in train_losses]
    val_list = [float(x) for x in val_losses]
    return (epochs, train_list, val_list)

# Step 23 - weights_l2_distance
import math

def weights_l2_distance(w_gd, w_closed):
    return math.sqrt(np.sum((w_gd - w_closed) ** 2))

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    return {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'seed': seed,
        'weights': None,
        'normal_weights': None,
        'mean': None,
        'std': None,
        'train_losses': [],
        'val_losses': []
    }

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    mean, std = compute_feature_stats(X_train)
    model['mean'] = mean
    model['std'] = std

    X_tr_dm = prepare_design_matrix(X_train, mean, std)
    X_va_dm = prepare_design_matrix(X_val, mean, std)

    model['normal_weights'] = normal_equation(X_tr_dm, y_train)

    best_weights, train_losses, val_losses = train_batch_gd(
        X_tr_dm, y_train, X_va_dm, y_val,
        lr=model['learning_rate'],
        epochs=model['epochs'], patience=model['patience'],
        seed=model['seed'])
    model['weights'] = best_weights
    # model['best_weights'] = best_weights
    model['train_losses'] = train_losses
    model['val_losses'] = val_losses
    return model

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

