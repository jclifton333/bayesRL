import numpy as np
import yaml
import os
import pdb

def read_files(policy_name, num_act):
  means = np.array([])
  standard_errors = np.array([])
  for fname in os.listdir():
    if policy_name in fname and 'numAct-{}'.format(num_act) in fname:
      results_dict = yaml.load(open(fname, 'rb'))
      means = np.append(means, results_dict['mean_regret'])
      standard_errors = np.append(standard_errors, results_dict['se_regret'])
  mean_ = np.mean(means)
  standard_error_ = np.sqrt(np.mean(standard_errors**2 * 48) / 192)
  return mean_, standard_error_
    
