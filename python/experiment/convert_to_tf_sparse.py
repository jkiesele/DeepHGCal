import os
import sys
from Queue import Queue # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
from small_calorimeter import convert

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def run_conversion(input_file):
    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'

    convert(input_file, os.path.join(args.output,just_file_name))


def worker():
    while True:
        input_file = jobs_queue.get()
        run_conversion(input_file)
        jobs_queue.task_done()


jobs_queue = Queue()
for i in range(int(args.jobs)):
     t = Thread(target=worker)
     t.daemon = True
     t.start()

for item in file_paths:
    jobs_queue.put(item)

jobs_queue.join()
