import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  for i in xrange(num_train):
      #print(i)
      numerator = np.exp(scores[i, y[i]])
      denominator = np.sum(np.exp(scores[i,:]))
      loss_i = 0.0
      for j in xrange(num_class):
          if j != y[i]:
              #print(-1 * np.exp(scores[i, y[i]]) / numerator / denominator \
                        #* (-1) * np.exp(scores[i,j]) * X[i])
              #print(dW[:,j])
              dW[:,j] += -1 / denominator \
                        * (-1) * np.exp(scores[i,j]) * X[i]
          else:
              dW[:,j] += -1 / denominator \
                        * X[i] * (np.sum(np.exp(scores[i, :])) - np.exp(scores[i, y[i]]))
      loss_i = -1 * np.log(numerator/denominator)
      loss += loss_i
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  '''
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  scores_exp = np.exp(scores)
  # denominators: (N, 1)
  denominators = np.sum(scores_exp, axis = 1)
  #print(denominators)
  # numerators: (N, 1)
  numerators = scores_exp[np.arange(num_train), y]
  #print(numerators)
  loss = -np.sum(np.log(numerators / denominators))

  weight_matrix = -scores_exp
  weight_matrix[np.arange(num_train), y] += np.sum(scores_exp, axis = 1)
  weight_matrix /= - denominators.reshape(-1,1)
  dW = X.T.dot(weight_matrix)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
  '''
  # using backpropagation
  # forward
  N = X.shape[0]
  C = W.shape[1]
  scores = X.dot(W)
  exps = np.exp(scores)
  den = np.sum(exps, axis = 1).reshape(-1,1)
  num = exps[np.arange(N), y].reshape(-1,1)
  invden = 1 / den
  frac = num * invden
  li = -np.log(frac)
  loss = np.sum(li) / N

  #backward

  dout = 1
  #loss = np.sum(li) / N
  dli = np.ones((N,1))*dout/N
  #li = -np.log(frac)
  dfrac = -1 / frac * dli
  #frac = num * invden
  dnum = invden * dfrac
  dinvden = num * dfrac
  #invden = 1 / den
  dden = -1 / (den ** 2)* dinvden
  #num = exps[np.arange(N), y].reshape(-1,1)
  dexps = np.zeros((N,C))
  dexps[np.arange(N), y] = dnum.reshape(1,-1)
  #den = np.sum(exps, axis = 1).reshape(-1,1)
  dexps += np.dot(dden, np.ones((1, C)))
  #exps = np.exp(scores)
  ds = np.exp(scores) * dexps

  #following up
  loss += reg * np.sum(W*W)
  dW = X.T.dot(ds)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
