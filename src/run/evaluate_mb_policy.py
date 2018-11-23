import sys
import pdb
import numpy as np
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.environments.Glucose import Glucose


def evaluate_glucose_mb_policy():
  # Roll out to get data
  n_patients = 10
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    pass

  # Fit model on data

  # Get optimal policy under model

  # Evaluate policy

  return