import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numpy as np
from time import time as getCurrentTime


def countFLOPS(model, width:int=1024, height:int=512)->int:
  """
    Counts the number of Floating Point Operations (FLOPs) in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count FLOPs for.
        width (int, optional): The width of the input image. Defaults to 1024.
        height (int, optional): The height of the input image. Defaults to 512.

    Returns:
        FLOPs (int): The total number of FLOPs in the model.

  """
  image = torch.zeros((1, 3, height, width)).cuda()

  return FlopCountAnalysis(model, image).total()


def latency(model, num_iterations:int=1000, width:int=1024, height:int=512)->list[float]:
  """
  Evaluates the latency and fps of a PyTorch model for a given number of iterations.

  Args:
      model (torch.nn.Module): The PyTorch model to evaluate.
      num_iterations (int, optional): The number of iterations to evaluate. Defaults to 1000.
      width (int, optional): The width of the input image. Defaults to 1024.
      height (int, optional): The height of the input image. Defaults to 512.

  Returns:
      latency and FPS (list[float]): The list of values
        - the mean latency of the model
        - the standard deviation of the latency of the model
        - the mean fps of the model
        - the standard deviation of the fps of the model
  """
  image = torch.randn(1, 3, height, width).cuda()

  latency, FPS = np.zeros(num_iterations), np.zeros(num_iterations)

  for i in range(num_iterations):
      latency[i] = getCurrentTime()

      _ = model(image)

      latency[i] = getCurrentTime() - latency[i]
      FPS[i] = 1/latency[i]

  return np.mean(latency)*1000, np.std(latency)*1000, np.mean(FPS), np.std(FPS)