import numpy as np
import yaml
import pdb


def get_operating_chars(fname):
    res = yaml.load(open(fname, 'rb'))
    num_tests = len(res['h1_true'][0])

    t1_errors = np.zeros(num_tests)
    total_t1_errors = np.zeros(num_tests)
    h0_counts = np.zeros(num_tests)
    all_counts = np.zeros(num_tests)
    for t1, h1 in zip(res['t1_errors'], res['h1_true']):
        h0_counter = 0
        for ix, h1_indicator in enumerate(h1):
            all_counts[ix] += 1
            if h1_indicator == 0:
                h0_counts[ix] += 1
                t1_errors[ix] += (t1[h0_counter] - t1_errors[ix]) / h0_counts[ix]
                total_t1_errors += (t1[h0_counter] - total_t1_errors[ix]) / all_counts[ix]
                h0_counter += 1
            else: 
              total_t1_errors[ix] += (0.0 - total_t1_errors[ix]) / all_counts[ix]
    return t1_errors, total_t1_errors
            

