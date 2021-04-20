"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """
  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object.
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    TODO:
    Implement initialization of the network.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super().__init__()
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # torch.nn.ReLU(inplace=False)
    self.conv1 = nn.Sequential(
                      nn.Conv2d(n_channels, 64, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(64),
                      nn.ReLU()
    )
    # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    self.maxpool1 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    self.conv2 =  nn.Sequential(
                      nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(128),
                      nn.ReLU()
    )
    self.maxpool2 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    self.conv3_a = nn.Sequential(
                      nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(256),
                      nn.ReLU()
    )
    self.conv3_b = nn.Sequential(
                      nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(256),
                      nn.ReLU()
    )
    self.maxpool3 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    self.conv4_a = nn.Sequential(
                      nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(512),
                      nn.ReLU()
    )
    self.conv4_b = nn.Sequential(
                      nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(512),
                      nn.ReLU()
    )
    self.maxpool4 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    self.conv5_a = nn.Sequential(
                      nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(512),
                      nn.ReLU()
    )
    self.conv5_b = nn.Sequential(
                      nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
                      nn.BatchNorm2d(512),
                      nn.ReLU()
    )
    self.maxpool5 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    # torch.nn.Linear(in_features, out_features, bias=True)
    self.linear = nn.Linear(512, n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################
  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    TODO:
    Implement forward pass of the network.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.maxpool1(self.conv1(x))
    out = self.maxpool2(self.conv2(out))
    out = self.maxpool3(self.conv3_b(self.conv3_a(out)))
    out = self.maxpool4(self.conv4_b(self.conv4_a(out)))
    out = self.maxpool5(self.conv5_b(self.conv5_a(out)))
    out = self.linear(out.squeeze())
    ########################
    # END OF YOUR CODE    #
    #######################
    return out