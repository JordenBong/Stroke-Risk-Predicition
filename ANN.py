import numpy as np

# Network architecture
input_size = 10
hidden_size = 10
output_size = 10  # The output layer before the final classification neuron


# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    """
    Initialize weights and biases for the neural network.

    Parameters:
    input_size -- Number of input nodes
    hidden_size -- Number of nodes in the hidden layer
    output_size -- Number of output nodes

    Returns:
    w_input_hidden -- Initialized weights for input-hidden layer
    b_hidden -- Initialized biases for hidden layer
    w_hidden_output -- Initialized weights for hidden-output layer
    b_output -- Initialized biases for output layer
    w_output_classify -- Initialized weights for output-classification neuron
    b_classify -- Initialized bias for classification neuron
    """
    np.random.seed(42)  # For reproducibility

    # Initialize weights with small random values
    w_input_hidden = np.random.randn(hidden_size, input_size) * 0.01
    w_hidden_output = np.random.randn(output_size, hidden_size) * 0.01
    w_output_classify = np.random.randn(1, output_size) * 0.01

    # Initialize biases as zeros
    b_hidden = np.zeros((hidden_size, 1))
    b_output = np.zeros((output_size, 1))
    b_classify = np.zeros((1, 1))

    return w_input_hidden, b_hidden, w_hidden_output, b_output, w_output_classify, b_classify


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(Z):
    return np.maximum(0, Z)


def forward_propagation(X, w_input_hidden, b_hidden, w_hidden_output, b_output, w_output_classify, b_classify):
    """
    Perform forward propagation through the neural network.

    Parameters:
    X -- Input data (features) of shape (input_size, batch_size)
    w_input_hidden -- Weights for input-hidden layer
    b_hidden -- Biases for hidden layer
    w_hidden_output -- Weights for hidden-output layer
    b_output -- Biases for output layer
    w_output_classify -- Weights for output-classification neuron
    b_classify -- Bias for classification neuron

    Returns:
    a_hidden -- Activations of the hidden layer after ReLU activation
    a_output -- Activations of the output layer before classification after ReLU activation
    a_classify -- Activations of the classification neuron after sigmoid activation
    """
    # Input to Hidden Layer
    z_hidden = np.dot(w_input_hidden, X) + b_hidden
    a_hidden = relu(z_hidden)  # Use ReLU activation

    # Hidden to Output Layer (before classification)
    z_output = np.dot(w_hidden_output, a_hidden) + b_output
    a_output = relu(z_output)

    # Output (before classification) to Classification Neuron
    z_classify = np.dot(w_output_classify, a_output) + b_classify
    a_classify = sigmoid(z_classify)

    return a_hidden, a_output, a_classify


def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)


def backpropagation(X, Y, a_hidden, a_output, a_classify,
                    w_input_hidden, w_hidden_output, w_output_classify):
    """
    Perform backpropagation to calculate gradients of the loss with respect to parameters.

    Parameters:
    X -- Input data (features) of shape (input_size, batch_size)
    Y -- True labels (ground truth) of shape (output_size, batch_size)
    a_hidden -- Activations of the hidden layer after forward propagation
    a_output -- Activations of the output layer before classification after forward propagation
    a_classify -- Activations of the classification neuron after forward propagation
    w_input_hidden -- Weights for input-hidden layer
    w_hidden_output -- Weights for hidden-output layer
    w_output_classify -- Weights for output-classification neuron

    Returns:
    dw_input_hidden -- Gradients of weights for input-hidden layer
    db_hidden -- Gradients of biases for hidden layer
    dw_hidden_output -- Gradients of weights for hidden-output layer
    db_output -- Gradients of biases for output layer
    dw_output_classify -- Gradients of weights for output-classification neuron
    db_classify -- Gradient of bias for classification neuron
    """
    # Compute gradients
    dz_classify = a_classify - Y
    dw_output_classify = np.dot(dz_classify, a_output.T) / X.shape[1]
    db_classify = np.sum(dz_classify, axis=1, keepdims=True) / X.shape[1]

    dz_output = np.dot(w_output_classify.T, dz_classify) * sigmoid_derivative(a_output)
    dw_hidden_output = np.dot(dz_output, a_hidden.T) / X.shape[1]
    db_output = np.sum(dz_output, axis=1, keepdims=True) / X.shape[1]

    dz_hidden = np.dot(w_hidden_output.T, dz_output) * relu_derivative(a_hidden)
    dw_input_hidden = np.dot(dz_hidden, X.T) / X.shape[1]
    db_hidden = np.sum(dz_hidden, axis=1, keepdims=True) / X.shape[1]

    return dw_input_hidden, db_hidden, dw_hidden_output, db_output, dw_output_classify, db_classify


def compute_loss(Y_true, Y_pred):
    """
    Compute binary cross-entropy loss.

    Parameters:
    Y_true -- true labels (ground truth)
    Y_pred -- predicted labels

    Returns:
    loss -- computed loss
    """
    m = Y_true.shape[1]
    loss = -1 / m * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    return loss


def compute_accuracy(Y_true, Y_pred):
    """
    Compute accuracy.

    Parameters:
    Y_true -- true labels (ground truth)
    Y_pred -- predicted labels

    Returns:
    accuracy -- computed accuracy
    """
    m = Y_true.shape[1]
    predictions = (Y_pred > 0.5).astype(int)
    accuracy = np.sum(predictions == Y_true) / m
    return accuracy


def update_parameters(w_input_hidden, b_hidden, w_hidden_output, b_output, w_output_classify, b_classify,
                      dw_input_hidden, db_hidden, dw_hidden_output, db_output, dw_output_classify, db_classify,
                      learning_rate):
    """
    Update parameters using gradient descent.

    Parameters:
    w_input_hidden -- weights for input-hidden layer
    b_hidden -- biases for hidden layer
    w_hidden_output -- weights for hidden-output layer
    b_output -- biases for output layer
    w_output_classify -- weights for output-classification neuron
    b_classify -- bias for classification neuron
    dw_input_hidden -- gradients of weights for input-hidden layer
    db_hidden -- gradients of biases for hidden layer
    dw_hidden_output -- gradients of weights for hidden-output layer
    db_output -- gradients of biases for output layer
    dw_output_classify -- gradients of weights for output-classification neuron
    db_classify -- gradient of bias for classification neuron
    learning_rate -- learning rate for gradient descent

    Returns:
    updated_w_input_hidden -- updated weights for input-hidden layer
    updated_b_hidden -- updated biases for hidden layer
    updated_w_hidden_output -- updated weights for hidden-output layer
    updated_b_output -- updated biases for output layer
    updated_w_output_classify -- updated weights for output-classification neuron
    updated_b_classify -- updated bias for classification neuron
    """
    updated_w_input_hidden = w_input_hidden - learning_rate * dw_input_hidden
    updated_b_hidden = b_hidden - learning_rate * db_hidden
    updated_w_hidden_output = w_hidden_output - learning_rate * dw_hidden_output
    updated_b_output = b_output - learning_rate * db_output
    updated_w_output_classify = w_output_classify - learning_rate * dw_output_classify
    updated_b_classify = b_classify - learning_rate * db_classify

    return updated_w_input_hidden, updated_b_hidden, updated_w_hidden_output, updated_b_output, updated_w_output_classify, updated_b_classify


def get_predictions(probs):
    return np.argmax(probs, 0)


class ANN():

    def __init__(self):
        # Initialize parameters
        (self.w_input_hidden, self.b_hidden, self.w_hidden_output, self.b_output, self.w_output_classify,
         self.b_classify) = initialize_parameters(input_size, hidden_size, output_size)

        # Hyperparameters
        self.learning_rate = 0.001
        self.num_epochs = 20

        # Dictionary to store weights
        self.Weights = {}
        # decay rate for decaying the learning rate over time
        self.decay_rate = 5

        # Lists to store accuracy, loss
        self.accuracy_list = []
        self.loss_list = []

    def fit(self, X_train, y_train):
        for epoch in range(self.num_epochs):
            # Forward propagation on the entire training set
            a_hidden, a_output, a_classify = forward_propagation(X_train.T, self.w_input_hidden, self.b_hidden,
                                                                 self.w_hidden_output, self.b_output,
                                                                 self.w_output_classify, self.b_classify)

            # Calculate loss
            loss = compute_loss(y_train.reshape(1, -1), a_classify)

            # Calculate accuracy
            accuracy = compute_accuracy(y_train.reshape(1, -1), a_classify)

            # Backpropagation
            dw_input_hidden, db_hidden, dw_hidden_output, db_output, dw_output_classify, db_classify = backpropagation(
                X_train.T, y_train.reshape(1, -1),
                a_hidden, a_output, a_classify, self.w_input_hidden, self.w_hidden_output, self.w_output_classify)

            learning_rate = (1 / (1 + self.decay_rate)) * self.learning_rate

            # Update parameters
            (self.w_input_hidden, self.b_hidden, self.w_hidden_output, self.b_output, self.w_output_classify,
             self.b_classify) = update_parameters(self.w_input_hidden, self.b_hidden, self.w_hidden_output,
                                                  self.b_output, self.w_output_classify, self.b_classify,
                                                  dw_input_hidden, db_hidden, dw_hidden_output,
                                                  db_output, dw_output_classify, db_classify, learning_rate)

            # Print accuracy and loss
            # if (epoch + 1) % 4 == 0:
            #     print(
            #         f"Epoch {epoch + 1}/{self.num_epochs} [==============================] - accuracy : "
            #         f"{accuracy:.4f} - loss : {loss:.4f}  - learning_rate : {learning_rate}")

            # Save accuracy and loss values
            self.accuracy_list.append(accuracy)
            self.loss_list.append(loss)

            # Saving the weights at the last epoch
            if epoch == self.num_epochs - 1:
                self.Weights[f'epoch_{epoch}'] = {
                    'w_input_hidden': self.w_input_hidden,
                    'b_hidden': self.b_hidden,
                    'w_hidden_output': self.w_hidden_output,
                    'b_output': self.b_output,
                    'w_output_classify': self.w_output_classify,
                    'b_classify': self.b_classify
                }

    def print_model_accuracy(self):
        print(f'The accuracy after training the neural network is {round(self.accuracy_list[-1] * 100, 2)}%')

    def make_predictions(self, X_test):
        _, _, probs = forward_propagation(X_test.T, self.w_input_hidden, self.b_hidden, self.w_hidden_output,
                                          self.b_output, self.w_output_classify, self.b_classify)
        predictions = get_predictions(probs)
        return probs, predictions

    def make_predictions_probability(self, X_test):
        _, _, probs = forward_propagation(X_test.T, self.w_input_hidden, self.b_hidden, self.w_hidden_output,
                                          self.b_output, self.w_output_classify, self.b_classify)
        return probs