import numpy as np

def fit(model, x : np.array, y : np.array, x_val:np.array = None, y_val:np.array = None, lr: float = 0.5, num_steps : int = 500):
    """
    Function to fit the logistic regression model using gradient ascent.

    Args:
        model: the logistic regression model.
        x: it's the input data matrix.
        y: the label array.
        x_val: it's the input data matrix for validation.
        y_val: the label array for validation.
        lr: the learning rate.
        num_steps: the number of iterations.

    Returns:
        history: the values of the log likelihood during the process.
    """
    # Initialize arrays to store history
    likelihood_history = []
    val_loss_history = []

    for it in range(num_steps):
        # Step 1: Make predictions
        preds = model.predict(x)

        # Step 2: Compute the gradient of the log-likelihood
        gradient = model.compute_gradient(x, y, preds)

        # Step 3: Update the model parameters using the gradient
        model.update_theta(gradient, lr)

        # Step 4: Compute log-likelihood and store it in history
        likelihood = model.likelihood(preds, y)
        likelihood_history.append(likelihood)


        if x_val is not None and y_val is not None:
            val_preds = model.predict(x_val)
            val_loss = -model.likelihood(val_preds, y_val)
            val_loss_history.append(val_loss)

    # Convert lists to numpy arrays before returning
    return np.array(likelihood_history), np.array(val_loss_history) if val_loss_history else np.array([])
