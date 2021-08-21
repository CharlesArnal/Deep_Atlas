#%load_ext tensorboard
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
import math
import pickle
import scipy.io as sio
import sklearn.datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import LinearSVC

from scipy.stats import iqr

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
# Used for the encoders and the decoders
# dim_layers is a list of integers describing the dimension of each layer (the output layer being last)
# activation_functions is a list of Tensorflow activation functions describing the activation function to be used for each layer
# Example: test_model = create_FFNN([4,5],[tf.nn.tanh,tf.exp])
def create_FFNN(dim_layers, activation_functions):
  model = tf.keras.Sequential()
  for layer_index,layer_dim in enumerate(dim_layers):
    model.add(tf.keras.layers.Dense(layer_dim, activation=activation_functions[layer_index]))
  return model

# Combination of an encoder and a decoder (both are FFNNs)
# Specify the dimension of the layers of the encoder, their activation functions, the dimensions of the layers of the decoder, and their activation functions
# (nonsensical) example: test_chart = Chart([4,5],[tf.nn.tanh,tf.tanh],[3,9,6],[tf.nn.tanh,tf.tanh,tf.exp])
class Chart(tf.keras.Model):
    def __init__(self, dim_layers_encoder, activation_functions_encoder,dim_layers_decoder, activation_functions_decoder, **kwargs):
        super(Chart, self).__init__(**kwargs)
        self.encoder = create_FFNN(dim_layers_encoder, activation_functions_encoder)
        self.decoder = create_FFNN(dim_layers_decoder, activation_functions_decoder)
        # We have to save those informations to easily make copies of the charts in the PCAE model
        self.dim_layers_encoder = dim_layers_encoder
        self.activation_functions_encoder = activation_functions_encoder
        self.dim_layers_decoder = dim_layers_decoder
        self.activation_functions_decoder = activation_functions_decoder

    def call(self, x):
      z = self.encode(x)
      x_reconstructed = self.decode(z)
      return [z,x_reconstructed]

    def encode(self,x):
      return self.encoder(x)

    def decode(self,z):
      return self.decoder(z)

    # Returns a new Chart that has the same weights (though they are not shared),
    # plus some small random noise
    # The Chart must have been built/used on some data before
    def create_copy_with_noise(self,noise_std = 0.025):
      copied_chart = Chart(self.dim_layers_encoder, self.activation_functions_encoder,self.dim_layers_decoder, self.activation_functions_decoder)
      input_dim = len(self.encoder.layers[0].get_weights()[0])
      copied_chart.build(input_shape = (None, input_dim))
      copied_chart.set_weights(self.get_weights())
      # copied_chart = deepcopy(self) DEEPCOPY DOES NOT WORK
      for weight in copied_chart.trainable_weights:
        weight.assign_add(tf.random.normal(weight.get_shape(),mean=0.0,stddev=noise_std))
      return copied_chart
    

"""
test_chart = Chart([4,5],[tf.nn.tanh,tf.tanh],[3,9,6],[tf.nn.tanh,tf.tanh,tf.exp])
test_chart.build([None,3])
test_chart.summary()
tf.print(test_chart.encoder.output_shape[-1])
"""

# The output layer should be a Dense layer with activation function None
# The rest_of_the_network can be anything (including None)
class Gating_network(tf.keras.Model):
    def __init__(self, output_layer, rest_of_the_network = None, **kwargs):
        super(Gating_network, self).__init__(**kwargs)
        self.output_layer = output_layer
        self.rest_of_the_network = rest_of_the_network

    def call(self, x):
      if self.rest_of_the_network is not None:
        y1 = self.rest_of_the_network(x)
        y = self.output_layer(y1)
      else:
        y = self.output_layer(x)
      return y

    # Splits an output unit into two copies of itself such that each 
    # gets part of the points currently assigned to that unit
    # The Gating_network should have been built before calling split_output_unit
    # TODO: make code slightly less ad-hoc?
    def split_output_unit(self, points_currently_assigned, unit_index, clustering_space = "encoding_space", clustering_alg = "k_means", encoding_points_currently_assigned = None, perturbation_scale = 0.06):
      [saved_mult_weights,saved_biases] = self.output_layer.get_weights()[-2:]
      previous_number_of_output_units = len(saved_biases)
      output_dim_rest_of_the_network = len(saved_mult_weights)
      
      # add a new output unit (copy of the unit of index unit_index)
      new_biases = np.append(saved_biases,saved_biases[unit_index])
      new_mult_weights = np.append(saved_mult_weights,np.array([saved_mult_weights[:,unit_index]]).transpose(), axis = 1)
      
      # Use a clustering algorithm either in the input space, in the encoding space, or in the codomain of self.rest_of_the_network
      if clustering_alg == "k_means":
        clusterer = KMeans(n_clusters=2, random_state=0)
      elif clustering_alg == "agglom_clustering":
        clusterer = AgglomerativeClustering(n_clusters=2,linkage="single")
      else:
        tf.print("Invalid clustering algorithm")
        return -1

      if self.rest_of_the_network is not None:
        nonlinear_mapping_of_the_points = self.rest_of_the_network(tf.convert_to_tensor(points_currently_assigned)).numpy()
      else:
        nonlinear_mapping_of_the_points = points_currently_assigned
        
      if clustering_space == "input_space":
        clusterer.fit(points_currently_assigned)
      elif clustering_space == "gating_network_encoding_space":
        clusterer.fit(nonlinear_mapping_of_the_points)
      elif clustering_space == "encoding_space":
        clusterer.fit(encoding_points_currently_assigned)
      # Separate using Linear SVM
      svc = LinearSVC()
      svc.fit(nonlinear_mapping_of_the_points,clusterer.labels_)
      # The separating hyperplane is of the shape hyperplane_vector *x + intercept = 0
      hyperplane_vector, intercept = svc.coef_[0], svc.intercept_
      # Renormalize
      magnitude_correction = perturbation_scale*tf.reduce_mean(tf.math.reduce_euclidean_norm(saved_mult_weights, axis = 0))/(np.linalg.norm(hyperplane_vector)+10**(-8))
      hyperplane_vector, intercept = hyperplane_vector * magnitude_correction, intercept*magnitude_correction
      # Add to the new unit the classification function from the SVM
      new_biases[-1] += intercept
      for i in range(output_dim_rest_of_the_network):
        new_mult_weights[i][-1] += hyperplane_vector[i]
      """ # legacy
      # add small noise to both copies
      new_biases[unit_index] += np.random.normal(loc=0.0,scale=noise_std)
      new_biases[-1] +=np.random.normal(loc=0.0,scale=noise_std)
      for i in range(output_dim_rest_of_the_network):
        new_mult_weights[i][unit_index] += np.random.normal(loc=0.0,scale=noise_std)
        new_mult_weights[i][-1] += np.random.normal(loc=0.0,scale=noise_std)
      if self.rest_of_the_network is not None:
        for weight in self.rest_of_the_network.trainable_weights:
          weight.assign_add(tf.random.normal(weight.get_shape(),mean=0.0,stddev=noise_std))
      """
      # create and save the new output_layer
      self.output_layer = tf.keras.layers.Dense(previous_number_of_output_units+1, activation = None )
      self.output_layer.build(input_shape = (None,output_dim_rest_of_the_network) )
      self.output_layer.set_weights([new_mult_weights,new_biases])
      return {"class0": [x for index, x in enumerate(points_currently_assigned) if clusterer.labels_[index] ==0 ],\
        "class1": [x  for index, x in enumerate(points_currently_assigned) if clusterer.labels_[index] == 1] }

"""
v = tf.constant([[0,2],[3,4],[-1,7]])
test1 = Gating_network(tf.keras.layers.Dense(2, activation=None),tf.keras.layers.Dense(3, activation=None))
#test1 = Gating_network(tf.keras.layers.Dense(2, activation=None),None)
test1(v)
tf.print(test1.output_layer.get_weights())
test1.split_output_unit(1,100)
tf.print(test1.output_layer.get_weights())
"""

class PCAE(tf.keras.Model):
  def __init__(self, list_of_charts=[],gating_network=None, main_loss_type = "exp", load_softmax_like = False,load_softmax_exponent =5, softmax_exponent = 1.0, alpha_load=0.1, alpha_importance= 0.1, alpha_classification = 0, alpha_inverse_reconstruction = 0, verbose = False, run_eagerly = False, **kwargs):
    super(PCAE, self).__init__(**kwargs)

    # Basic elements:
    #-----------
    # Must be a list of Charts objects
    self.list_of_charts = list_of_charts
    # Must be a Gating_network object
    self.gating_network = gating_network

    # Variations:
    #-----------
    # Allows for a more extreme softmax (the larger it is, the more weight the largest values have)
    self.softmax_exponent = softmax_exponent
    # Small variation during training:
    # if "outside", the loss is \sum_i \alpha_i ||\psi_i(\phi_i(X))-X||^2
    # if "inside", the loss is ||\sum_i \alpha_i \psi_i(\phi_i(X))-X||^2
    # if "exp", the loss is -log(\sum_i \alpha_i exp(-||\psi_i(\phi_i(X))-X||^2))
    # (where alpha_i are the weights of the charts)
    self.main_loss_type = main_loss_type
    # Different way to compute the load loss
    # Rmk: load_softmax_exponent needs to be large (at least 3 or 4), as it is used
    # on softmax_weights (that are between 0 and 1)
    self.load_softmax_like = load_softmax_like
    self.load_softmax_exponent = load_softmax_exponent
    # Weights for different losses
    # alpha_inverse_reconstruction and alpha_classification can only be non 0 if
    # training is run in eager mode
    self.alpha_load = alpha_load
    self.alpha_importance = alpha_importance
    self.alpha_inverse_reconstruction = alpha_inverse_reconstruction
    self.alpha_classification = alpha_classification

    # if verbose, also computes the sublosses that were not used in the total loss
    self.verbose = verbose
    # approximates versions of the inverse reconstruction and classification losses
    # are computed if not in eager mode
    self.run_eagerly = run_eagerly


  def call(self, x, training=False):
    # x of shape [batch_size, dim_input]
    # The gating network outputs a tensor of shape [batch_size,num_charts]
    # which gives a weight to each chart (for a given point)
    weights_charts = self.gating_network(x)
    if training:
      # Modified softmax; softmax_weights of shape [batch_size, num_charts]
       # TODO : check if ok
      renormalized_weights_charts = weights_charts - tf.stop_gradient(tf.reduce_max(weights_charts, axis =-1, keepdims = True))
      softmax_weights = tf.math.divide(tf.exp(renormalized_weights_charts*self.softmax_exponent) , tf.expand_dims(tf.reduce_sum(tf.exp(renormalized_weights_charts*self.softmax_exponent), axis = -1), -1) )
      # batch_size = x.get_shape().as_list()
      # z is only used for the inverse reconstruction loss in training mode
      z_output = []
      x_reconstructed_output = []
      for chart_index, chart in enumerate(self.list_of_charts):
        # z and x_reconstructed of shape [batch_size, dim_x/z]
        [z,x_reconstructed] = chart(x)
        x_reconstructed_output.append(x_reconstructed)
        z_output.append(z)
        # x_reconstructed_output and z_output are lists (of size num_charts) of tensors of shape [batch_size, dim_x/z]
      return [z_output, softmax_weights, x_reconstructed_output]
    else:
      # For each point of the batch, we find the index of the chart with the largest weight
      # index_chart is of shape [batch_size]
      indices_chart = tf.math.argmax(weights_charts,1)
      z_output = []
      x_reconstructed_output = []
      # The loop goes over each point of the batch
      # TODO check how that impacts performance; it should be possible to parallelize this,
      # or at least do it in a more elegant way
      for point_in_batch,index_chart in enumerate(indices_chart):
        # Encode each point of the batch using the encoder of the selected chart,
        # then decode it using its decoder
        # tf.expand_dims(x[point_in_batch,:], axis=0) is of shape [1,dim_x]
        [z,x_reconstructed] = self.list_of_charts[index_chart](tf.expand_dims(x[point_in_batch,:], axis=0))
        # z and x_reconstructed are of shape [1,dim_x/z] - we squeeze them to get tensors of thape [dim_x/z]
        z_output.append(tf.squeeze(z, axis = 0))
        x_reconstructed_output.append(tf.squeeze(x_reconstructed, axis=0))
      # Turn lists of tensors of shape [dim_z] (resp. [dim_x]) into tensors of shape
      # [batch_size,dim_z] (resp. [batch_size,dim_x])
      z_output = tf.stack(z_output,0)
      x_reconstructed_output = tf.stack(x_reconstructed_output,0)
      return [z_output,indices_chart,x_reconstructed_output]

  def train_step(self, x):
    # TODO Would it be better to use keras.losses. ...?
    with tf.GradientTape() as tape:
      total_loss = 0
      [z, softmax_weights, x_reconstructed] =  self(x,training= True)     
      main_reconstruction_loss = self.main_reconstruction_loss(x,softmax_weights,x_reconstructed,self.main_loss_type)
      total_loss += main_reconstruction_loss
      losses_tracking = {"main_reconstruction_loss": main_reconstruction_loss}
      losses_tracking["number_of_charts"] = len(self.list_of_charts)
      if self.verbose or self.alpha_load != 0:
        load_loss =  self.load_loss(softmax_weights)
        losses_tracking["load_loss"] = load_loss
        if self.alpha_load != 0:
          total_loss += self.alpha_load*load_loss
      if self.verbose or self.alpha_importance != 0:
        importance_loss =  self.importance_loss(softmax_weights)
        losses_tracking["importance_loss"] = importance_loss
        if self.alpha_importance != 0:
          total_loss += self.alpha_importance*importance_loss
      if self.verbose or self.alpha_inverse_reconstruction != 0:
        inverse_reconstruction_loss =  self.inverse_reconstruction_loss(tf.stop_gradient(z),x_reconstructed,tf.stop_gradient(softmax_weights), run_eagerly = self.run_eagerly)
        losses_tracking["inverse_reconstruction_loss"] = inverse_reconstruction_loss
        if self.alpha_inverse_reconstruction != 0:
          total_loss += self.alpha_inverse_reconstruction *inverse_reconstruction_loss
      if self.verbose or self.alpha_classification != 0:
        classification_loss = self.classification_loss(x_reconstructed, tf.stop_gradient(softmax_weights),run_eagerly = self.run_eagerly)
        losses_tracking["classification_loss"] = classification_loss
        if self.alpha_classification != 0:
          total_loss += self.alpha_classification*classification_loss
    # IMPORTANT! batch size must be quite large for load_loss and importance_loss to make sense
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    # TODO track losses in a more professional way
    return losses_tracking

  # The main loss function 
  # Computes the squared distance between x and the weighted average of the tensors of x_reconstructed
  # if main_loss_type is "inside", the weighted average of the squared distance of x and
  # the tensors of x_reconstructed if it is "outside", and a formula with an exp and a log if it is "exp"
  def main_reconstruction_loss(self,x,softmax_weights,list_of_x_reconstructed,main_loss_type):
    if main_loss_type == "inside":
      # Turns list_of_x_reconstructed_output (which is a list of tensors) 
      # into a tensor of shape [batch_size,dim_x,num_charts]
      x_reconstructed = tf.stack(list_of_x_reconstructed,-1)
      # Weighted sum (by softmax_weights) of the reconstructed outputs of each chart.
      # Now x_reconstructed_output is of shape [batch_size,dim_x]
      x_reconstructed = tf.linalg.matvec(x_reconstructed,softmax_weights)
      return tf.reduce_mean(tf.reduce_sum(tf.math.square(x-x_reconstructed),axis=1),0)
    elif main_loss_type == "outside":
      errors = []
      for x_reconstructed in list_of_x_reconstructed:
        errors.append(x-x_reconstructed)
      # Turns errors (which is a list of tensors) 
      # into a tensor of shape [batch_size,dim_x,num_charts]
      errors = tf.stack(errors,-1)
      # squared_errors of shape [batch_size,num_charts]
      squared_errors = tf.reduce_sum(tf.math.square(errors),1)
      # weighted_squared_errors of shape [batch_size]
      weighted_squared_errors = tf.reduce_sum(tf.multiply(squared_errors,softmax_weights), -1)
      return tf.reduce_mean(weighted_squared_errors)
    elif main_loss_type == "exp":
      errors = []
      for x_reconstructed in list_of_x_reconstructed:
        errors.append(x-x_reconstructed)
      # Turns errors (which is a list of tensors) 
      # into a tensor of shape [batch_size,dim_x,num_charts]
      errors = tf.stack(errors,-1)
      # squared_errors of shape [batch_size,num_charts]
      squared_errors = tf.reduce_sum(tf.math.square(errors),1)
      exp_errors = tf.math.exp(-0.5*squared_errors)
      # weighted_exp_errors of shape [batch_size]
      weighted_exp_errors = tf.reduce_sum(tf.multiply(exp_errors,softmax_weights), -1)
      return tf.reduce_mean(-tf.math.log(weighted_exp_errors))
  
  # Must be called in eager mode
  # x_reconstructed and z are lists (of length num_charts) of tensors of shape [batch_size, dim_x/z]
  # softmax_weights of shape [batch_size, num_charts]
  # The loss cannot be exactly computed in graph mode, in which case we compute an approximation of it
  def inverse_reconstruction_loss(self,z,x_reconstructed,softmax_weights, run_eagerly = False):
    if run_eagerly:
      # dominant_chart_indices of shape [batch_size]
      dominant_chart_indices = tf.math.argmax(softmax_weights,1)
      errors = []
      for sample, index in enumerate(dominant_chart_indices):
        # computes the image of x_reconstructed by the encoder of the relevant chart (need to be a bit cunning with the shape of the tensors)
        z_reconstructed = self.list_of_charts[index].encoder(tf.expand_dims(x_reconstructed[index][sample], axis =0))
        errors.append(z[index][sample]-tf.squeeze(z_reconstructed, axis = 0))
      # errors is a list of length batch_size of tensors of shape [dim_z]
      # this turns it into a tensor of shape [batch_size, dim_z]
      errors = tf.stack(errors)
      loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(errors),-1))
      return loss
    else:
      squared_errors = []
      for index_chart, batch_of_x in enumerate(x_reconstructed):
        # error of shape [batch_size, dim_z]
        error = z[index_chart]-self.list_of_charts[index_chart].encoder(batch_of_x)
        # squared_errors is a list of length num_charts of tensors of shape [batch_size]
        squared_errors.append(tf.reduce_sum(tf.square(error),-1))
      # turns squared_errors into a tensor of shape [batch_size,num_charts]
      squared_errors = tf.stack(squared_errors,-1)
      # putting the softmax_weights to the power 7 should make the dominant one VERY dominant
      modified_softmax_weights = tf.math.divide(softmax_weights**10 , tf.expand_dims(tf.reduce_sum(softmax_weights**10, axis = -1), -1) )
      weighted_squared_errors = tf.reduce_sum(tf.multiply(squared_errors,modified_softmax_weights), -1)
      loss = tf.reduce_mean(weighted_squared_errors)
      return loss
      
    
  # Must be called in eager mode
  # x_reconstructed is a list (of length num_charts) of tensors of shape [batch_size, dim_x]
  # softmax_weights of shape [batch_size, num_charts]
  def classification_loss(self,x_reconstructed, softmax_weights, run_eagerly = False):
    if run_eagerly:
      # dominant_chart_indices of shape [batch_size]
      dominant_chart_indices = tf.math.argmax(softmax_weights,1)
      classification_errors =[]
      for sample, index in enumerate(dominant_chart_indices):
        # new_weights of shape [1,num_charts]
        new_weights = self.gating_network(tf.expand_dims(tf.stop_gradient(x_reconstructed[index][sample]), axis =0))
        new_weights = tf.squeeze(new_weights, axis = 0)
        new_renormalized_weights = new_weights - tf.stop_gradient(tf.reduce_max(new_weights, axis =-1, keepdims = True))
        new_softmax_weights = tf.math.divide(tf.exp(new_renormalized_weights*self.softmax_exponent) , tf.expand_dims(tf.reduce_sum(tf.exp(new_renormalized_weights*self.softmax_exponent), axis = -1), -1) )
        classification_errors.append(-tf.math.log(new_softmax_weights[index]))
      classification_errors = tf.stack(classification_errors)
      loss = tf.reduce_mean(classification_errors)
      return loss
    else:
      classification_errors = []
      for chart_index, batch_of_x in enumerate(x_reconstructed):
        # new_weights of shape [batch_size,num_charts]
        new_weights = self.gating_network(tf.stop_gradient(batch_of_x))
        new_renormalized_weights = new_weights - tf.stop_gradient(tf.reduce_max(new_weights, axis =-1, keepdims = True))
        new_softmax_weights = tf.math.divide(tf.exp(new_renormalized_weights*self.softmax_exponent) , tf.expand_dims(tf.reduce_sum(tf.exp(new_renormalized_weights*self.softmax_exponent), axis = -1), -1) )
        # classification_errors is a list of length num_charts of tensors of shape [batch_size]
        classification_errors.append(-tf.math.log(new_softmax_weights[:,chart_index]))
      # classification_errors is now a tensor of shape [batch_size,num_charts]
      classification_errors = tf.stack(classification_errors,-1)
      modified_old_softmax_weights = tf.math.divide(softmax_weights**15 , tf.expand_dims(tf.reduce_sum(softmax_weights**15, axis = -1), -1) )
      #weighted_classification_errors of shape [batch_size]
      weighted_classification_errors = tf.reduce_sum(tf.multiply(classification_errors,modified_old_softmax_weights), -1)
      loss = tf.reduce_mean(weighted_classification_errors)
      return loss


  # Helps evenly distribute the samples among the charts 
  def importance_loss(self,softmax_weights):
    # softmax_weights of shape [batch_size, num_charts]
    # importance of shape [num_charts]
    importance = tf.reduce_sum(softmax_weights, axis= 0)
    CV_importance = tf.math.square(tf.math.reduce_std (importance)/(tf.reduce_mean(importance)+10**(-8)))
    return CV_importance

  # Helps evenly distribute the samples among the charts 
  # softmax_weights of shape [batch_size, num_charts]
  def load_loss(self,softmax_weights):
    if self.load_softmax_like == False:
      num_charts = softmax_weights.get_shape().as_list()[-1]
      # max_weights of dim [batch_size, 1]
      max_weights = tf.math.reduce_max(softmax_weights, axis = -1, keepdims = True)
      # renormalized_differences of dim [batch_size, num_charts]
      renormalized_differences = (softmax_weights - tf.stop_gradient(max_weights))*15*num_charts
      # TODO: maybe replace by another function that has the same main properties,
      # but which is easier to compute?
      # Defines a Normal distribution
      dist = tfd.Normal(loc=0., scale=1.)
      # load of shape [num_charts]
      load = tf.reduce_sum(dist.cdf(renormalized_differences),axis = 0) 
    else:
      # Use a new exponent for a softmax-like formula
      # modified_softmax_weights of shape [batch_size, num_charts]
      # TODO : check that no problem
      """
      renormalized_softmax_weights = softmax_weights - tf.stop_gradient(tf.reduce_max(softmax_weights, axis =-1, keepdims = True))
      modified_softmax_weights = tf.math.divide(tf.exp(renormalized_softmax_weights*self.load_softmax_exponent) , tf.expand_dims(tf.reduce_sum(tf.exp(renormalized_softmax_weights*self.load_softmax_exponent), axis = -1), -1) )      
      """ # There was a mistake
      modified_softmax_weights = tf.math.divide(softmax_weights**self.load_softmax_exponent , tf.expand_dims(tf.reduce_sum(softmax_weights**self.load_softmax_exponent, axis = -1), -1) )      
      # load of shape [num_charts]
      load = tf.reduce_sum(modified_softmax_weights, axis = 0)
    CV_load = tf.math.square(tf.math.reduce_std (load)/(tf.reduce_mean(load)+10**(-7)))
    return CV_load

  # Code is a bit awkward, and not very Tensorflow friendly
  # TODO: add criterion on minimal load (number of points assigned) of chart being split?
  # TODO: check that the variance of the random perturbation on the new map is reasonable
  # possible values for clustering_space: "encoding_space", "gating_network_encoding_space", "input_space"
  # clustering_alg takes values in "k_means" or "agglom_clustering"
  def create_new_chart(self,x,error_tolerance, clustering_space = "encoding_space", clustering_alg = "k_means", average_or_total = "average"):
    num_charts = len(self.list_of_charts)
    batch_size = tf.shape(x)[0]
    noise_std = 0.025
    magnitude_change_gating_network = 0.05
    # x and x_reconstructed of shape [batch_size, dim_x]
    [z,indices_charts,x_reconstructed] = self(x, training = False)
    # average_chart_errors of shape [num_charts]
    average_chart_errors = np.zeros([num_charts],dtype= np.float32)
    # chart_load = tf.zeros([num_charts])
    charts_loads = [ len([index for index in indices_charts if index == i]) for i in range(num_charts)]
    for sample_index in range(batch_size):
      if average_or_total == "average":
        average_chart_errors[indices_charts[sample_index]] += tf.reduce_sum(tf.math.square(x[sample_index,:]-x_reconstructed[sample_index,:]))/(charts_loads[indices_charts[sample_index]]+10**(-8))
      else:
        average_chart_errors[indices_charts[sample_index]] += tf.reduce_sum(tf.math.square(x[sample_index,:]-x_reconstructed[sample_index,:]))
    # TODO meditate on best criterion for detecting underperforming charts
    if tf.math.reduce_max(average_chart_errors) > error_tolerance\
    or tf.math.reduce_max(average_chart_errors) > tf.reduce_mean(average_chart_errors) + 1.5* iqr(average_chart_errors, interpolation="linear"):
      # find the worst performing chart
      index_chart_to_split = tf.math.argmax(average_chart_errors)
      # create a slightly modified copy of this chart
      self.list_of_charts.append(self.list_of_charts[index_chart_to_split].create_copy_with_noise(noise_std))
      # slightly modify the original chart
      for weight in  self.list_of_charts[index_chart_to_split].trainable_weights:
        weight.assign_add(tf.random.normal(weight.get_shape(),mean=0.0,stddev=noise_std))
      # identify the data points currently assigned to the original chart
      points_assigned_to_chart = np.array([ point.numpy() for index, point in enumerate(x) if indices_charts[index] == index_chart_to_split ])
      # Split the associated output unit of the gating network
      encoding_points_assigned_to_chart = np.array([ encoded_point.numpy() for index, encoded_point in enumerate(z) if indices_charts[index] == index_chart_to_split ])
      classes_split = self.gating_network.split_output_unit(points_assigned_to_chart,index_chart_to_split, clustering_space = clustering_space, clustering_alg = clustering_alg, encoding_points_currently_assigned = encoding_points_assigned_to_chart, perturbation_scale = magnitude_change_gating_network)
      tf.print("New chart created")
      return classes_split
    else:
      return None
     


# IMPORTANT: make sure to specify an output dimension for your charts that is equal to the dimension of your input
