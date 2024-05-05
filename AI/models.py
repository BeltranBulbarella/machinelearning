import nn
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        Should compute the dot product of the stored weight vector and the given input, returning an nn.DotProduct object.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        return 1 if the dot product is non-negative or −1
        use nn.as_scalar to convert a scalar Node into a Python floating-point number.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Should repeatedly loop over the data set and make updates on examples that are misclassified.
        Use the update method of the nn.Parameter class to update the weights.
        When an entire pass over the data set is completed without making any mistakes, 100% training accuracy has been achieved, and training can terminate.
        """
        while True:
            mistake = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    mistake = True
            if not mistake:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.W1 = nn.Parameter(1, 40)  # First layer of weights
        self.b1 = nn.Parameter(1, 40)  # Bias of the first layer
        self.W2 = nn.Parameter(40, 1)  # Second layer of weights
        self.b2 = nn.Parameter(1, 1)  # Bias of the second layer

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x = nn.Linear(x, self.W1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.W2)
        x = nn.AddBias(x, self.b2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Tests:
                learning_rate: 0.02
                epochs: 100
                Your final loss (0.089931) must be no more than 0.0200 to receive full points for this question

                learning_rate: 0.01
                epochs: 1000
                Your final loss (0.025888) must be no more than 0.0200 to receive full points for this question

                learning_rate: 0.01
                epochs: 2000
                Your final loss (0.075072) must be no more than 0.0200 to receive full points for this question

                learning_rate: 0.005
                epochs: 1000
                Your final loss (0.072677) must be no more than 0.0200 to receive full points for this question

                Got it changing the model parameters
        """

        learning_rate = 0.01 # The learning rate
        epochs = 1000 # The number of epochs, an epoch is a complete pass through the dataset
        for epoch in range(epochs):
            total_loss = 0
            num_examples = 0
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y) # Compute the loss for this example
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2 = nn.gradients(loss,
                                                                                  [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(grad_wrt_W1, -learning_rate) # Update the weights, why? to minimize the loss
                self.b1.update(grad_wrt_b1, -learning_rate)
                self.W2.update(grad_wrt_W2, -learning_rate)
                self.b2.update(grad_wrt_b2, -learning_rate)

                total_loss += nn.as_scalar(loss)
                num_examples += 1
            average_loss = total_loss / num_examples
            print(f'Epoch {epoch + 1}, Loss: {average_loss}')
            if average_loss <= 0.02:  # Stop training if the loss is small enough
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        self.W1 = nn.Parameter(784, 300)
        self.b1 = nn.Parameter(1, 300)
        self.W2 = nn.Parameter(300, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x = nn.Linear(x, self.W1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.W2)
        x = nn.AddBias(x, self.b2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = 0.01
        epochs = 5
        for epoch in range(epochs):
            total_loss = 0
            num_examples = 0
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2 = nn.gradients(loss,
                                                                                  [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(grad_wrt_W1, -learning_rate)
                self.b1.update(grad_wrt_b1, -learning_rate)
                self.W2.update(grad_wrt_W2, -learning_rate)
                self.b2.update(grad_wrt_b2, -learning_rate)

                total_loss += nn.as_scalar(loss)
                num_examples += 1
            average_loss = total_loss / num_examples
            print(f'Epoch {epoch + 1}, Loss: {average_loss}')


class LanguageIDModel(object):
    def __init__(self):
        self.num_chars = 47  # Number of unique characters across all languages
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_size = 400

        # Input weights for the first character
        self.W_initial = nn.Parameter(self.num_chars, self.hidden_size)
        # Recurrent weights for subsequent characters
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        # Bias for the hidden state
        self.b_hidden = nn.Parameter(1, self.hidden_size)
        # Weights for the output layer
        self.W_output = nn.Parameter(self.hidden_size, len(self.languages))
        # Bias for the output layer
        self.b_output = nn.Parameter(1, len(self.languages))
        # Additional hidden layer
        self.W_hidden2 = nn.Parameter(self.hidden_size, self.hidden_size)  # Second recurrent layer weights
        self.b_hidden2 = nn.Parameter(1, self.hidden_size)  # Second recurrent layer bias

    def run(self, xs):
        """
        Run the model for a batch of examples.
        xs is a list of length L of nodes each of shape (batch_size x self.num_chars)
        """
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W_initial), self.b_hidden))

        for x in xs[1:]:
            h = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x, self.W_initial), nn.Linear(h, self.W_hidden)), self.b_hidden))

        logits = nn.AddBias(nn.Linear(h, self.W_output), self.b_output)
        return logits

    def get_loss(self, xs, y):
        """
        Compute the softmax loss of the batch of examples.
        xs: list of input character nodes
        y: true labels node of shape (batch_size x 5)
        """
        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Train the model on the provided dataset.
        """
        learning_rate = 0.01
        epochs = 50
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            for xs, y in dataset.iterate_once(100):  # Assuming batch size of 100
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss,
                                         [self.W_initial, self.W_hidden, self.b_hidden, self.W_output, self.b_output])

                # Update each parameter
                self.W_initial.update(gradients[0], -learning_rate)
                self.W_hidden.update(gradients[1], -learning_rate)
                self.b_hidden.update(gradients[2], -learning_rate)
                self.W_output.update(gradients[3], -learning_rate)
                self.b_output.update(gradients[4], -learning_rate)

                total_loss += nn.as_scalar(loss)
                count += 1

            average_loss = total_loss / count
            print("Epoch {}: Loss = {}".format(epoch + 1, average_loss))

        print("Training complete.")