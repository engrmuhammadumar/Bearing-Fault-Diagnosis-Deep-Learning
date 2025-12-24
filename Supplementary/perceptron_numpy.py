import numpy as np

def relu(z):
    return np.maximum(0, z)

def neuron(x, w, b):
    """
    x: input vector, shape (n,)
    w: weight vector, shape (n,)
    b: bias (scalar)
    """
    z = np.dot(w, x) + b   # weighted sum
    y = relu(z)            # activation
    return y

if __name__ == "__main__":
    # Example input
    x = np.array([200.0, -100.0])   # [x1, x2]
    w = np.array([0.5, 1.0])    # [w1, w2]
    b = -0.5

    y = neuron(x, w, b)
    print("Neuron output:", y)
