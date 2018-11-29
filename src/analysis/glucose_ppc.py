import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import yaml
import numpy as np
import matplotlib.pyplot as plt


def ppc_for_one_step_q_functions():
  """
  Compare ppd of one-step q functions fit to NP conditional glucose estimator to model-free one-step q function.

  :return:
  """
  # Get saved data
  mf_fname = os.path.join(project_dir, 'run', 'results', 'glucose-mf-vfn_181128_203831.yml')
  mb_fname = os.path.join(project_dir, 'run', 'results', 'glucose-vfn-ppd_181128_194407.yml')
  mf = yaml.load(open(mf_fname, 'r'))
  mb = yaml.load(open(mb_fname, 'r'))
  v_mf = mf['v_mf']
  v_mb = np.array(mb['v_mf'])  # mb data was given incorrect key

  # Plot at a few points;
  # The value functions were evaluated on a grid where glucose varied acc to np.linspace(50, 200, 100) (and the other
  # values were fixed at their observed means).  So e.g. v_mf[0] gives the estimated mf value function at
  # glucose = 50.

  plt.hist(v_mb[:, 0])
  plt.vlines(v_mf[0])
  plt.show()

  return


if __name__ == "__main__":
  ppc_for_one_step_q_functions()




