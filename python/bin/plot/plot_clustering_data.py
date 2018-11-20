import os
import sys
from queue import Queue  # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import sparse_hgcal
from libs.merge import merge_two_arrays, merge_two_arrays_separate
import time
from libs.prepare_merge_set import load_data_target_center

max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
args = parser.parse_args()

# X = -150 to  150
# Y = -150 to  150
# Z =    0 to 2000


limits_x = (0,0)
limits_y = (0,0)
limits_z = (0,0)
def iterate(input_file):
    global limits_x, limits_y, limits_z
    return_dict = load_data_target_center(input_file)
    data_output = return_dict['data_output_merged']
    num_entries_result = return_dict['num_entries_result']
    data_output_unmerged = return_dict['data_output_unmerged_energies']
    target_x = data_output[:,:,-3]
    target_y = data_output[:,:,-2]
    target_z = data_output[:,:,-1]
    limits_x = min(limits_x[0], np.min(target_x)), max(limits_x[0], np.max(target_x))
    limits_y = min(limits_y[0], np.min(target_y)), max(limits_y[0], np.max(target_y))
    limits_z = min(limits_z[0], np.min(target_z)), max(limits_z[0], np.max(target_z))


iterate(args.input)

print("Limits X", limits_x)
print("Limits Y", limits_y)
print("Limits Z", limits_z)