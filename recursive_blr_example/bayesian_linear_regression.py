#!/usr/bin/env python2.7
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import gamma
from numpy.linalg import inv
import warnings
from copy import deepcopy


class NIGParams:
  """
  Class to handle prior parameters
  """
  w = None
  V = None
  a = None
  b = None

  def __init__(self, w=None, V=None, a=None, b=None):
    self.w = w
    self.V = V
    self.a = a
    self.b = b

  def get(self):
    return self.w, self.V, self.a, self.b

  def __str__(self):
    return "w: " + str(self.w) + \
           " V: " + str(self.V) + \
           " a: " + str(self.a) + \
           " b: " + str(self.b)


class BayesianLinearRegression:

  prior_params = None
  post_params = None

  def __init__(self):
    self.prior_params = NIGParams()
    self.post_params = NIGParams()
    return

  def set_prior_params(self, w_0, V_0, a_0, b_0):
    """
    Directly set the parameters of the prior
    :param w_0: mean weights
    :param V_0: V_0^-1 is analogous to X'X, or the covariance of the inputs
                (appears as V_0^-1 + X'X)
    :param a_0: shape parameter of the Inv Gamma distribution over var. This is
                half the strength of the prior (appears as a_0 + n/2)
    :param b_0: rate parameter of the Inv Gamma. It is n_0 * sigma_0^2.
    :return:
    """
    self.prior_params = NIGParams(w=w_0, V=V_0, a=a_0, b=b_0)

  def set_prior_from_data(self, X, Y, weights=None, n=1):
    """
    Set the prior parameters from the empirical
    :param X: inputs from prior-generating data set
    :param Y: outputs from prior-generating data set
    :param weights: optional weights for training data-set
    :param n: number of inputs in prior-generating data-set. Defaults to 1
    :return: None
    """
    n_examples = Y.size
    assert Y.shape[0] == n_examples, "Y must be (n_examples x 1)"
    assert X.shape[0] == n_examples, "X must be (n_examples x dimension)"

    if weights is None:
      weights = np.ones_like(Y)
    else:
      assert weights.shape[0] == n_examples, "weights must be (n_examples x 1)"
      if n_examples > 0:
        assert weights.shape[1] == 1, "weights must be (n_examples x 1)"

    # make diagonal matrix for weights to simplify math expression
    W = np.diag(weights.flatten()) * n / n_examples

    # Calculate the posterior parameters
    V_N = inv(np.transpose(X).dot(W.dot(X)))
    w_N = V_N.dot(np.transpose(X).dot(W.dot(Y)))
    a_N = np.trace(W)/2.
    b_N = 0.5 * (np.transpose(Y).dot(W.dot(Y)) - np.transpose(w_N).dot(inv(V_N).dot(w_N)))

    # save params
    self.set_prior_params(w_0=w_N, V_0=V_N, a_0=a_N, b_0=b_N)
    return


  def IG(self, x, a, b):
    """
    Inverse Gamma distribution
    :param x: test point
    :param a: shape parameter (n/2)
    :param b: rate parameter (sigma^2*n)
    :return: p(x|a,b) for Inverse Gamma distribution
    """
    return b ** a / gamma(a) * x ** (-a - 1) * np.exp(-b / x)

  def fit_posterior(self, X, Y, weights=None):
    """
    Fit posterior parameters given inputs, X, outputs, Y, and weights, W
    :param X: inputs (n_examples x dimensions) array
    :param Y: outputs (n_examples x 1) array
    :param weights: (n_examples x 1) array
    :return: None
    """
    n_examples = Y.size
    assert Y.shape[0] == n_examples, "Y must be (n_examples x 1)"
    assert X.shape[0] == n_examples, "X must be (n_examples x dimension)"

    if weights is None:
      weights = np.ones_like(Y)
    else:
      assert weights.shape[0] == n_examples, "weights must be (n_examples x 1)"
      if n_examples > 0:
        assert weights.shape[1] == 1, "weights must be (n_examples x 1)"

    # If no data, set the posterior as the prior
    if n_examples == 0:
      self.post_params = self.prior_params
      return

    # load prior params for convenience
    V_0 = self.prior_params.V
    w_0 = self.prior_params.w
    a_0 = self.prior_params.a
    b_0 = self.prior_params.b

    if (V_0 is None) or (w_0 is None) or (a_0 is None) or (b_0 is None):
      warnings.warn("Fitting params without prior!")
      self.set_prior_from_data(X, Y, None, Y.size)
      self.post_params = deepcopy(self.prior_params)
      return

    # make diagonal matrix for weights to simplify math expression
    W = np.diag(weights.flatten())

    # Calculate the posterior parameters
    V_N = inv(inv(V_0) + np.transpose(X).dot(W.dot(X)))
    w_N = V_N.dot(inv(V_0).dot(w_0) + np.transpose(X).dot(W.dot(Y)))
    a_N = a_0 + np.trace(W)/2.
    b_N = b_0 + 0.5 * (np.transpose(w_0).dot(inv(V_0).dot(w_0)) + np.transpose(Y).dot(W.dot(Y)) - np.transpose(w_N).dot(inv(V_N).dot(w_N)))

    # save posterior params
    self.post_params = NIGParams(w=w_N, V=V_N, a=a_N, b=b_N)

    return

  def pred_posterior(self, X):
    """
    generate predictions for new data using posterior params
    :param X: new input data. Each row must be one training point.
    :return: mean, standard deviation for prediction
    """
    # generate predictions for Bayesian with unknown noise variance
    w_N, V_N, a_N, b_N = self.post_params.get()
    if a_N <= 2:
      warnings.warn("a_N must be greater than 2! Returning Normal approximation to Student t distribution (may be optimistic)", RuntimeWarning)
      mu_pred = X.dot(w_N)
      sig_pred = np.sqrt(b_N / a_N * (np.ones_like(mu_pred) + np.diag(X.dot(V_N.dot(np.transpose(X)))).reshape([-1, 1])))
    else:
      mu_pred = X.dot(w_N)
      sig_pred = np.sqrt(2 * a_N / (2 * a_N - 2) * float(b_N) / a_N * (np.ones_like(mu_pred) + np.diag(X.dot(V_N.dot(np.transpose(X)))).reshape([-1, 1])))

    return mu_pred, sig_pred

  def pred_prior(self, X):
    """
    generate predictions for new data using prior params
    :param X: new input data. Each row must be one training point.
    :return: mean, standard deviation for prediction
    """
    # generate predictions for Bayesian with unknown noise variance
    w, V, a, b = self.prior_params.get()
    if a <= 2:
      warnings.warn("a must be greater than 2! Returning Normal approximation to Student t distribution (may be optimistic)", RuntimeWarning)
      mu_pred = X.dot(w)
      sig_pred = np.sqrt(b / a * (np.ones_like(mu_pred) + np.diag(X.dot(V.dot(np.transpose(X)))).reshape([-1, 1])))
    else:
      mu_pred = X.dot(w)
      sig_pred = np.sqrt(2 * a / (2 * a - 2) * b / a * (np.ones_like(mu_pred) + np.diag(X.dot(V.dot(np.transpose(X)))).reshape([-1, 1])))

    return mu_pred, sig_pred

  def get_prior_params(self):
    """
    return prior mean and covariance of the weights and the
    variance of the prediction
    """
    w_0, V_0, a_0, b_0 = self.prior_params.get()
    Sigma_w_0 = b_0 / a_0 * V_0
    var_y_0 = b_0 / (a_0 - 1)
    return w_0, Sigma_w_0, var_y_0

  def get_posterior_params(self):
    """
    return posterior mean and covariance of the weights and the
    variance of the prediction
    """
    w_N, V_N, a_N, b_N = self.post_params.get()
    Sigma_w_N = b_N / a_N * V_N
    var_y_N = b_N / (a_N - 1)
    return w_N, Sigma_w_N, var_y_N

  def reweight_prior(self, new_prior_strength=None):
    """
    Re-compute the parameters of the prior so that it is as if the prior was
    derived from a data set with the same unit sufficient statistics but
    fewer samples. Should not increase the strength of the prior this way.
    That is like introducing artificial data which will lead to over-confident
    results
    :param new_prior_strength: New (smaller) effective sample size of the prior.
                               Defaults to keeping strength constant.
    """
    if new_prior_strength is None:
      new_prior_strength = 2. * self.prior_params.a

    if new_prior_strength > 2. * self.prior_params.a:
      warnings.warn("Received inflated value for re-weighting the prior. "
                    "The prior should only be re-weighted to a lower value!")

    a_N = self.prior_params.a
    b_N = self.prior_params.b
    w_N = self.prior_params.w
    V_N = self.prior_params.V

    self.prior_params.a = new_prior_strength / 2.
    self.prior_params.b = b_N / a_N * (new_prior_strength / 2.)
    self.prior_params.w = w_N
    self.prior_params.V = a_N / (new_prior_strength / 2.) * V_N

    return

  def set_posterior_to_prior(self):
    self.post_params = deepcopy(self.prior_params)
    return


  def update_recursive(self, X, Y):
    """
    Fit the posterior to a new batch of data but re-weight to keep the 
    effective number of points in the model the same.
    Also updates the prior to match.
    
    :param      X:    n_examples x dimension feature fector
    :type       X:    numpy array
    :param      Y:    n_examples x 1 target vector
    :type       Y:    numpy array
    """
    # fit posterior to new batch of data
    prior_strength = np.round(self.prior_params.a*2)
    self.fit_posterior(X, Y, None)

    # set the prior to the posterior
    self.prior_params = deepcopy(self.post_params)

    # re-weight the prior to get constant strength
    self.reweight_prior(new_prior_strength=prior_strength)

    # copy prior to posterior for prediction
    self.post_params = deepcopy(self.prior_params)
    return