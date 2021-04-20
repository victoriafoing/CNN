from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 50
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  TODO:
  Implement accuracy computation.
  """
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()
  ########################
  # END OF YOUR CODE    #
  #######################
  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.
  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # Choose to run on GPU or CPU
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # Get params
  lr = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq
  data_dir = FLAGS.data_dir

  # Load data
  cifar_data = cifar10_utils.get_cifar10(data_dir)
  train = cifar_data["train"]
  test = cifar_data["test"]

  # Define neural network
  net = ConvNet(train.images.shape[1], train.labels.shape[1]).to(device)

  # Define loss
  criterion = nn.CrossEntropyLoss()

  # Define optimizer
  optimizer = optim.Adam(net.parameters(), lr=lr)

  # Store losses
  train_losses = []
  test_losses = []

  # Store accuracies
  train_accuracies = []
  test_accuracies = []

  # Train
  for epoch in tqdm(range(max_steps)):

      # Select batch
      inputs, labels = train.next_batch(batch_size)

      # Convert to tensor
      inputs = torch.tensor(inputs, dtype=torch.float, device=device)
      labels = torch.tensor(labels, dtype=torch.long, device=device)

      # zero parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net.forward(inputs)
      loss = criterion(outputs, np.argmax(labels,axis=1))
      loss.backward()
      optimizer.step()

      # Print statistics
      if epoch % eval_freq == 0 or epoch == max_steps - 1:
          test_inputs, test_labels = test.next_batch(batch_size)
          test_inputs = torch.tensor(test_inputs, dtype=torch.float, device=device)
          test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)

          with torch.no_grad():
              test_outputs = net.forward(test_inputs)

          # Calculate losses
          train_loss = criterion(outputs, np.argmax(labels,axis=1)).item()
          train_losses.append(train_loss)
          test_loss = criterion(test_outputs, np.argmax(test_labels,axis=1)).item()
          test_losses.append(test_loss)

          # Calculate accuracies
          train_acc = accuracy(outputs, labels)
          train_accuracies.append(train_acc)
          test_acc = accuracy(test_outputs, test_labels)
          test_accuracies.append(test_acc)
          print("Epoch {}, train loss: {}, train acc: {}, test loss: {}, test acc: {}".format(epoch, round(train_loss, 2), train_acc, round(test_loss, 2), test_acc))
          ########################
          # END OF YOUR CODE    #
          #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()
  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)
  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  main()