import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    if n > 0 and isinstance(n, int):
        A=np.random.random([n, 1])
    return A
    raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    if h > 0 and w > 0 and type(h)==int and type(w)==int:
        A = np.random.rand(h, w)
        B = np.random.rand(h, w)
        s = A+B
        return A, B, s
    
    raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    s = np.linalg.norm(A + B)
    return s
    raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    nn = np.tanh(weights.T.dot(inputs))
    return nn
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x <= y:
        sf = x*y
    else:
        sf = x/y
    return sf
    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    vf = np.vectorize(scalar_function(x,y))
    return vf
    raise NotImplementedError
    
"""-----------------------------------------------------------------------------"""
Randomization = randomization(3)
print('Randomization = ')
print(Randomization)
print('------------------------------------------------------------------------')
Operations = operations(2, 2)
print('Operations = ')
print(Operations)
print('------------------------------------------------------------------------')
A = np.random.rand(2, 2)
B = np.random.rand(2, 2)
Norm = norm(A, B)
print('Norm = ')
print(Norm)
print('------------------------------------------------------------------------')
inputs = np.random.rand(2, 1)
weights = np.random.rand(2, 1)
Neural_Network = neural_network(inputs, weights)
print('Neural Network = ')
print(Neural_Network)
print('------------------------------------------------------------------------')
Scalar_Function = scalar_function(2, 2)
print('Scalar Function = ')
print(Scalar_Function)
print('------------------------------------------------------------------------')
Vector_Function = vector_function(2, 2)
print('Vector Function = ')
print(Vector_Function)
print('------------------------------------------------------------------------')