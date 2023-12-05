import torch
import os
import re

def getDevice():
  is_cuda = torch.cuda.is_available()
  return "cuda:0" if is_cuda else "cpu"