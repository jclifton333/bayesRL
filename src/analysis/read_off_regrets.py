import yaml
import numpy as np


if __name__ == '__main__':
  filenames = ['regrets-for-1-initial-pulls_181005_013413.yml', 'regrets-for-5-initial-pulls_181005_013417.yml',
               'regrets-for-15-initial-pulls_181005_013420.yml',
               'regrets-for-25-initial-pulls_181005_013423.yml',
               'regrets-for-35-initial-pulls_181005_012202.yml',
               'regrets-for-45-initial-pulls_181005_012311.yml',
               'regrets-for-55-initial-pulls_181005_012315.yml',
               'regrets-for-65-initial-pulls_181005_012206.yml']
  for fname in filenames:
    res = yaml.load(open(fname))
    regrets = [d['cumulative_regret'] for d in res]
    print(np.mean(regrets), np.std(regrets) / np.sqrt(len(regrets)))
