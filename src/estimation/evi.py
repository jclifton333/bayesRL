"""
Simulate from prior predictive distribution; update posterior; simulate from posterior predictive distribution;
do Q-learning; average against prior to compute expected value of information.
"""
import numpy as np


def two_stage_prior_predictive_draws(n_patients, x1_para, x2_para, y_para, x1_np, x2_np, y_np, alpha_x1, alpha_x2,
                                     alpha_y, policy1, policy2):
  """

  :param policy2:
  :param policy1:
  :param n_patients:
  :param x1_para: Prior predictive dbn for x1 under parametric model
  :param x2_para:
  :param y_para:
  :param x1_np:
  :param x2_np:
  :param y_np:
  :param alpha_x1:
  :param alpha_x2:
  :param alpha_y:
  :return:
  """
  # Draw from x1 prior pd
  x1 = []
  for patient in range(n_patients):
    if np.random.random() < alpha_x1:
      x1_draw = x1_np()
    else:
      x1_draw = x1_para()
    x1.append(x1_draw)

  action1 = policy1(x1)
  # Draw from conditional x2 prior pd
  x2 = []
  for patient in range(n_patients):
    x1_patient = x1[patient]
    action1_patient = action1[patient]
    if np.random.random() < alpha_x2:
      x2_draw = x2_np(x1_patient, action1_patient)
    else:
      x2_draw = x2_para(x1_patient, action1_patient)
    x2.append(x2)

  action2 = policy2(x1, action1, x2)
  # Draw from conditional y prior pd
  y = []
  for patient in range(n_patients):
    x1_patient = x1[patient]
    action1_patient = action1[patient]
    x2_patient = x2[patient]
    action2_patient = action2[patient]
    if np.random.random() < alpha_y:
      y_draw = y_np(x1_patient, action1_patient, x2_patient, action2_patient)
    else:
      y_draw = y_para(x1_patient, action1_patient, x2_patient, action2_patient)
    y.append(y_draw)

  return {'x1': x1, 'action1': action1, 'x2': x2, 'action2': action2, 'y': y}


def update_posteriors(x1, action1, x2, action2, y, pm_x1_para, pm_x1_np):
  """
  Update posterior based on output from two-stage.

  :param x1:
  :param action1:
  :param x2:
  :param action2:
  :param y:
  :param pm_para: pymc3 model parametric to be updated
  :param pm_np:   pymc3 model np to be updated
  :return:
  """
  # ToDo: Fill in
  return {'pm_para': pm_para, 'pm_np': pm_np}






